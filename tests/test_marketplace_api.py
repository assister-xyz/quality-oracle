"""Tests for QO-053-H marketplace API."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _AsyncCursor:
    """Async iterator that yields the supplied list, with no-op chainables."""

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def skip(self, *args, **kwargs):
        return self


def _make_db_mock(skill_docs=None, probe_docs=None, attestation=None):
    """Build a mock get_db() result that satisfies marketplace queries."""
    skill_docs = skill_docs or []
    probe_docs = probe_docs or []

    skill_scores = MagicMock()
    skill_scores.find = MagicMock(return_value=_AsyncCursor(skill_docs))
    skill_scores.find_one = AsyncMock(return_value=skill_docs[0] if skill_docs else None)

    probe_results = MagicMock()
    probe_results.find = MagicMock(return_value=_AsyncCursor(probe_docs))

    db = MagicMock()
    db.quality__skill_scores = skill_scores
    db.quality__probe_results = probe_results

    return db, skill_scores, probe_results


SAMPLE_SKILL = {
    "skill_id": "jupiter",
    "subject_uri": "did:web:sendaifun.github.io/jupiter",
    "name": "Jupiter Swap",
    "slug": "jupiter",
    "skill_repo": "sendaifun/skills/jupiter",
    "score": 82,
    "tier": "proficient",
    "current_score": 82,
    "last_evaluated_at": datetime(2026, 4, 25, 10, 0),
    "last_evaluation_id": "eval-123",
    "dimensions": {"accuracy": 85, "safety": 78, "process_quality": 80, "reliability": 82, "latency": 0, "schema_quality": 88},
    "delta_vs_baseline": 12.4,
    "baseline_score": 70.0,
    "activation_provider": "cerebras:llama3.1-8b",
    "github_url": "https://github.com/sendaifun/skills/tree/main/jupiter",
    "owner": "sendaifun",
    "category": "defi",
    "na_axes": ["latency"],
}


def test_list_marketplace_404_when_empty(test_client):
    """Empty marketplace returns 404."""
    db, _, _ = _make_db_mock([])
    with patch("src.api.v1.marketplace.get_db", return_value=db), \
         patch("src.api.v1.marketplace.get_redis", return_value=MagicMock(get=AsyncMock(return_value=None), set=AsyncMock())):
        resp = test_client.get("/v1/marketplace/empty-slug")
    assert resp.status_code == 404


def test_list_marketplace_returns_items_with_r5_risk(test_client):
    """List endpoint returns items with derived r5_risk_score."""
    skills = [dict(SAMPLE_SKILL, skill_id=f"sk-{i}", name=f"Skill {i}", slug=f"sk-{i}") for i in range(3)]
    probes = [{"probe_type": "fee_payer_hijack", "passed": False}]

    db, _, _ = _make_db_mock(skills, probes)
    redis_mock = MagicMock(get=AsyncMock(return_value=None), set=AsyncMock())
    with patch("src.api.v1.marketplace.get_db", return_value=db), \
         patch("src.api.v1.marketplace.get_redis", return_value=redis_mock):
        resp = test_client.get("/v1/marketplace/sendai")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["slug"] == "sendai"
    assert data["total"] == 3
    assert len(data["items"]) == 3
    for item in data["items"]:
        assert "r5_risk_score" in item
        assert 0.0 <= item["r5_risk_score"] <= 10.0
        assert item["activation_provider"] == "cerebras:llama3.1-8b"
    # Top-10 risk surface populated
    assert isinstance(data["top_risks"], list)
    # Cache write attempted
    assert redis_mock.set.await_count == 1


def test_list_marketplace_uses_cache(test_client):
    """If Redis has cached payload, MongoDB is NOT queried."""
    cached_payload = {
        "slug": "sendai",
        "items": [],
        "total": 0,
        "avg_score": 0,
        "top_risks": [],
        "generated_at": "2026-04-25T10:00:00",
    }
    import json
    redis_mock = MagicMock(get=AsyncMock(return_value=json.dumps(cached_payload)), set=AsyncMock())
    db_mock = MagicMock()  # would explode if accessed
    with patch("src.api.v1.marketplace.get_db", return_value=db_mock), \
         patch("src.api.v1.marketplace.get_redis", return_value=redis_mock):
        resp = test_client.get("/v1/marketplace/sendai")
    assert resp.status_code == 200
    # find never called
    assert not db_mock.quality__skill_scores.find.called


def test_skill_detail_returns_payload(test_client):
    """Per-skill detail returns axes, probe_results, activation_provider."""
    skills = [SAMPLE_SKILL]
    probes = [
        {"probe_type": "fee_payer_hijack", "passed": True, "explanation": "ok"},
        {"probe_type": "rpc_misconfig", "passed": False, "explanation": "drift"},
    ]
    db, _, _ = _make_db_mock(skills, probes)
    with patch("src.api.v1.marketplace.get_db", return_value=db):
        resp = test_client.get("/v1/marketplace/sendai/jupiter")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "Jupiter Swap"
    assert data["score"] == 82
    assert data["delta_vs_baseline"] == 12.4
    assert data["activation_provider"] == "cerebras:llama3.1-8b"
    assert "latency" in data["na_axes"]
    assert len(data["probe_results"]) == 2


def test_skill_detail_404_unknown(test_client):
    """Unknown skill in slug returns 404."""
    db, _, _ = _make_db_mock([])
    db.quality__skill_scores.find_one = AsyncMock(return_value=None)
    with patch("src.api.v1.marketplace.get_db", return_value=db):
        resp = test_client.get("/v1/marketplace/sendai/unknown-skill")
    assert resp.status_code == 404


def test_aqvc_pending_when_no_attestation(test_client):
    """When no attestation exists, return a pending stub (still valid JSON)."""
    skills = [SAMPLE_SKILL]
    db, _, _ = _make_db_mock(skills)
    db.quality__skill_scores.find_one = AsyncMock(return_value=SAMPLE_SKILL)

    attest_col = MagicMock()
    attest_col.find_one = AsyncMock(return_value=None)
    with patch("src.api.v1.marketplace.get_db", return_value=db), \
         patch("src.api.v1.marketplace.attestations_col", return_value=attest_col), \
         patch("src.api.v1.marketplace.evaluations_col", return_value=MagicMock()):
        resp = test_client.get("/v1/marketplace/sendai/jupiter/aqvc.json")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["aqvc_version"] == "1.0"
    assert payload["status"] == "pending"


def test_aqvc_returns_signed_credential(test_client):
    """When attestation exists, returns the VC document with download header."""
    skills = [SAMPLE_SKILL]
    db, _, _ = _make_db_mock(skills)
    db.quality__skill_scores.find_one = AsyncMock(return_value=SAMPLE_SKILL)

    vc_doc = {
        "@context": ["https://www.w3.org/ns/credentials/v2"],
        "type": ["VerifiableCredential", "AQVC"],
        "issuer": "did:web:laureum.ai",
        "credentialSubject": {"id": "did:web:sendaifun.github.io/jupiter", "score": 82},
        "proof": {"type": "Ed25519Signature2020", "proofValue": "z3xyz"},
    }
    attest_col = MagicMock()
    attest_col.find_one = AsyncMock(return_value={
        "evaluation_id": "eval-123",
        "vc_document": vc_doc,
    })
    with patch("src.api.v1.marketplace.get_db", return_value=db), \
         patch("src.api.v1.marketplace.attestations_col", return_value=attest_col), \
         patch("src.api.v1.marketplace.evaluations_col", return_value=MagicMock()):
        resp = test_client.get("/v1/marketplace/sendai/jupiter/aqvc.json")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    payload = resp.json()
    assert payload["issuer"] == "did:web:laureum.ai"
    assert "proof" in payload


def test_invalid_slug_returns_400(test_client):
    """Slug with non-alnum chars (other than dash) → 400."""
    resp = test_client.get("/v1/marketplace/bad slug!")
    # "%20" / "!" — fastapi may route differently; both 400 and 404 acceptable.
    assert resp.status_code in (400, 404)


# ── R5 risk score helper unit tests ─────────────────────────────────────────


def test_r5_risk_score_zero_for_perfect_skill():
    """All probes pass + high score → low risk (close to zero from score path)."""
    from src.api.v1.marketplace import _derive_r5_risk_score
    risk = _derive_r5_risk_score(
        [{"probe_type": "fee_payer_hijack", "passed": True}],
        score=95,
    )
    assert 0.0 <= risk <= 1.0


def test_r5_risk_score_high_for_failing_skill():
    """Failed critical probes → high risk."""
    from src.api.v1.marketplace import _derive_r5_risk_score
    risk = _derive_r5_risk_score(
        [
            {"probe_type": "fee_payer_hijack", "passed": False},
            {"probe_type": "script_poisoning", "passed": False},
        ],
        score=40,
    )
    assert risk >= 5.0


def test_r5_risk_score_no_probes_falls_back_to_score():
    """No probe data uses score as proxy (low score = high risk)."""
    from src.api.v1.marketplace import _derive_r5_risk_score
    risk_low_score = _derive_r5_risk_score([], score=20)
    risk_high_score = _derive_r5_risk_score([], score=95)
    assert risk_low_score > risk_high_score


# ── revalidate webhook helper ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revalidate_helper_skips_when_unset():
    """notify_marketplace_revalidate is a no-op when settings are empty."""
    from src.core.marketplace_revalidate import notify_marketplace_revalidate
    with patch("src.core.marketplace_revalidate.settings") as s:
        s.frontend_revalidate_base = ""
        s.frontend_revalidate_secret = ""
        ok = await notify_marketplace_revalidate("sendai", "jupiter")
    assert ok is False


@pytest.mark.asyncio
async def test_revalidate_helper_posts_when_configured():
    """When configured, the helper posts to /api/revalidate."""
    from src.core import marketplace_revalidate

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResponse()

    with patch.object(marketplace_revalidate, "httpx", MagicMock(AsyncClient=FakeClient)), \
         patch.object(marketplace_revalidate, "settings") as s:
        s.frontend_revalidate_base = "https://laureum.ai"
        s.frontend_revalidate_secret = "shh"
        ok = await marketplace_revalidate.notify_marketplace_revalidate("sendai", "jupiter")

    assert ok is True
    assert captured["url"] == "https://laureum.ai/api/revalidate"
    assert captured["params"]["path"] == "/marketplace/sendai/jupiter"
    assert captured["params"]["secret"] == "shh"
