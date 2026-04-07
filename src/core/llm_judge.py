"""
LLM-as-Judge for AgentTrust evaluation scoring.

Ported from agent-poi hackathon (poi/llm_judge.py).
Adapted for AgentTrust: OpenAI primary, DeepSeek fallback, Groq fallback2, fuzzy fallback.
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

CACHE_TTL = 86400  # 24 hours
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# Per-provider rate limits (requests per minute)
_PROVIDER_RPM = {
    "groq": 20,       # Free tier: 30 RPM, conservative headroom
    "openai": 40,     # Tier 1: ~60 RPM for gpt-4o-mini
    "deepseek": 40,
    "cerebras": 20,
    "gemini": 15,     # Free tier is lower
    "openrouter": 40,
    "mistral": 40,
}


class _ProviderRateLimiter:
    """Global per-provider rate limiter for LLM API calls."""

    _instances: Dict[str, "_ProviderRateLimiter"] = {}

    def __init__(self, provider: str, rpm: int):
        self.provider = provider
        self.min_gap = 60.0 / rpm  # seconds between calls
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    @classmethod
    def for_provider(cls, provider: str) -> "_ProviderRateLimiter":
        if provider not in cls._instances:
            rpm = _PROVIDER_RPM.get(provider, 30)
            cls._instances[provider] = cls(provider, rpm)
        return cls._instances[provider]

    async def wait(self):
        async with self._lock:
            now = time.time()
            wait_time = self._last_call + self.min_gap - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.time()

    async def backoff(self, seconds: float = 60.0):
        """Force a longer wait after sustained rate limiting."""
        async with self._lock:
            logger.info(f"Rate limit backoff: waiting {seconds:.0f}s for {self.provider}")
            self._last_call = time.time() + seconds - self.min_gap
            await asyncio.sleep(seconds)

# Stop words that carry no signal when comparing expected descriptions to JSON
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "that", "this", "these", "those", "it", "its", "if", "then",
    "than", "when", "where", "how", "what", "which", "who", "whom",
    "all", "each", "every", "any", "some", "no", "other",
})

# Verbs from expected-behavior descriptions that never appear in JSON responses
_BEHAVIOR_VERBS = frozenset({
    "return", "returns", "returned", "compute", "computes", "computed",
    "calculate", "calculates", "calculated", "handle", "handles", "handled",
    "provide", "provides", "provided", "contain", "contains", "contained",
    "include", "includes", "included", "show", "shows", "displayed",
    "gracefully", "correctly", "properly", "appropriately", "successfully",
    "validate", "validates", "validated", "convert", "converts", "converted",
    "fetch", "fetches", "fetched", "retrieve", "retrieves", "retrieved",
    "respond", "responds", "indicate", "indicates",
})

# Regex for extracting key='value' or key="value" patterns from expected text
_KV_PATTERN = re.compile(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]")

# Error indicator substrings
_ERROR_INDICATORS = ("error", "exception", "traceback", "validation error", "field required")


def _classify_answer(answer: str) -> str:
    """Classify an answer as 'json', 'error', or 'text'."""
    stripped = answer.strip()
    # Try JSON first
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    # Check for error indicators
    lower = stripped.lower()
    for indicator in _ERROR_INDICATORS:
        if indicator in lower:
            return "error"
    return "text"


def _extract_json_values(data, prefix: str = "") -> Dict[str, str]:
    """Recursively flatten JSON into {path: str(value)} pairs."""
    result = {}
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                result.update(_extract_json_values(v, path))
            else:
                result[path] = str(v)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                result.update(_extract_json_values(item, path))
            else:
                result[path] = str(item)
    return result


def _normalize_numeric(value: str) -> str:
    """Normalize numeric strings: '15.0' → '15', '3.00' → '3'."""
    try:
        num = float(value)
        if num == int(num):
            return str(int(num))
        return f"{num:.6g}"
    except (ValueError, TypeError):
        return value.strip().lower()


def _filter_content_terms(text: str) -> List[str]:
    """Extract meaningful content terms from text, filtering stop words."""
    # Extract key=value patterns and add their components
    kv_terms = []
    for match in _KV_PATTERN.finditer(text):
        kv_terms.append(match.group(1).lower())
        kv_terms.append(match.group(2).lower())

    # Remove kv patterns from text before splitting
    cleaned = _KV_PATTERN.sub("", text)
    # Split and filter
    words = re.split(r"[\s,.:;!?()]+", cleaned.lower())
    terms = [
        w.strip("'\"") for w in words
        if len(w) > 2
        and w.strip("'\"") not in _STOP_WORDS
        and w.strip("'\"") not in _BEHAVIOR_VERBS
    ]
    return kv_terms + terms


def _score_json_response(expected: str, answer: str) -> Optional[Tuple[int, str]]:
    """Score a JSON response against expected behavior text.

    Returns (score, explanation) or None if JSON parsing fails.
    """
    try:
        data = json.loads(answer.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    flat = _extract_json_values(data)
    json_values = [v.lower() for v in flat.values()]
    json_keys = [k.lower().split(".")[-1] for k in flat.keys()]
    expected_lower = expected.lower()

    # --- Signal 1: Parameter echo (35%) ---
    # Check if key='value' pairs from expected text appear in JSON values
    kv_matches = _KV_PATTERN.findall(expected_lower)
    if kv_matches:
        echo_hits = 0
        for _key, value in kv_matches:
            val_lower = _normalize_numeric(value)
            if any(_normalize_numeric(jv) == val_lower or val_lower in jv for jv in json_values):
                echo_hits += 1
        echo_score = echo_hits / len(kv_matches)
    else:
        # No explicit kv pairs - check if any expected content terms appear in values
        content_terms = _filter_content_terms(expected)
        if content_terms:
            hits = sum(1 for t in content_terms if any(t in jv for jv in json_values))
            echo_score = min(1.0, hits / max(len(content_terms), 1))
        else:
            echo_score = 0.5  # Neutral if no terms to check

    # --- Signal 2: Key coverage (30%) ---
    # Extract data-carrying nouns from expected, check against JSON keys
    content_terms = _filter_content_terms(expected)
    if content_terms:
        key_hits = 0
        for term in content_terms:
            # Substring match: "temperature" matches "temperature_c"
            if any(term in jk or jk in term for jk in json_keys):
                key_hits += 1
        key_score = min(1.0, key_hits / max(len(content_terms), 1))
    else:
        key_score = 0.5

    # --- Signal 3: Structural validity (35%) ---
    has_error_key = any("error" in k for k in json_keys) or any("error" in v for v in json_values)
    error_expected = any(
        kw in expected_lower
        for kw in ("error", "fail", "invalid", "missing", "gracefully", "reject")
    )

    if has_error_key and error_expected:
        struct_score = 0.85
    elif has_error_key and not error_expected:
        struct_score = 0.15  # Unexpected error
    elif len(flat) >= 4:
        struct_score = 1.0  # Rich response
    elif len(flat) >= 2:
        struct_score = 0.8
    else:
        struct_score = 0.6

    raw = echo_score * 0.35 + key_score * 0.30 + struct_score * 0.35
    score = max(0, min(100, int(round(raw * 100))))

    return score, (
        f"JSON: echo={echo_score:.0%}, keys={key_score:.0%}, struct={struct_score:.0%}"
    )


def _score_error_response(expected: str, answer: str) -> Tuple[int, str]:
    """Score an error response against expected behavior."""
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    error_expected = any(
        kw in expected_lower
        for kw in ("error", "fail", "invalid", "missing", "gracefully", "reject", "exception")
    )

    if error_expected:
        # Error was expected - base score + detail bonus
        base = 60
        detail_bonus = 0
        detail_keywords = ["required", "missing", "invalid", "validation", "type", "field", "parameter"]
        for kw in detail_keywords:
            if kw in answer_lower:
                detail_bonus += 4
        detail_bonus = min(25, detail_bonus)
        score = min(100, base + detail_bonus)
        return score, f"Error expected+received, detail_bonus={detail_bonus}"
    else:
        # Unexpected error
        return 15, "Unexpected error in response"


# Test types that can be reliably scored by the fuzzy scorer without LLM.
# These produce structured error messages, type errors, or boundary violations
# where keyword/pattern matching is sufficient.
FUZZY_ROUTABLE_TEST_TYPES = frozenset({
    "error_handling",
    "type_coercion",
    "boundary",
})


@dataclass
class JudgeResult:
    """Result from the LLM judge evaluation."""
    score: int  # 0-100
    explanation: str
    method: str  # "llm", "fuzzy", or "fuzzy_routed"
    cached: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""


@dataclass
class CacheEntry:
    result: JudgeResult
    timestamp: float


class _KeyRotator:
    """Round-robin API key rotator for providers with multiple keys."""

    def __init__(self, keys_csv: str):
        self._keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        self._index = 0
        self._exhausted: set = set()  # keys that returned quota errors

    @property
    def current(self) -> Optional[str]:
        available = [k for k in self._keys if k not in self._exhausted]
        if not available:
            return None
        return available[self._index % len(available)]

    def reset_exhausted(self):
        """Reset exhausted keys (call between evaluations or after cooldown)."""
        self._exhausted.clear()

    def rotate(self, exhausted: bool = False):
        """Rotate to next key. If exhausted=True, mark current key as rate-limited."""
        if exhausted and self.current:
            self._exhausted.add(self.current)
        self._index += 1

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def available_count(self) -> int:
        return len([k for k in self._keys if k not in self._exhausted])


@dataclass
class JudgeMetrics:
    """Tracks cost optimization and token usage metrics for a single evaluation session."""
    llm_calls: int = 0
    fuzzy_routed: int = 0
    cache_hits: int = 0
    total_judged: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_provider: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record_tokens(self, provider: str, input_tokens: int, output_tokens: int):
        """Record token usage for a provider."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if provider not in self.by_provider:
            self.by_provider[provider] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        self.by_provider[provider]["input_tokens"] += input_tokens
        self.by_provider[provider]["output_tokens"] += output_tokens
        self.by_provider[provider]["calls"] += 1

    def summary(self) -> str:
        saved = self.fuzzy_routed + self.cache_hits
        pct = f"{saved / self.total_judged * 100:.0f}%" if self.total_judged else "0%"
        tokens_str = f", tokens: {self.total_input_tokens}in/{self.total_output_tokens}out"
        return (
            f"Judge metrics: {self.total_judged} total, "
            f"{self.llm_calls} LLM, {self.fuzzy_routed} fuzzy-routed, "
            f"{self.cache_hits} cached ({pct} LLM calls saved)"
            f"{tokens_str}"
        )

    def to_dict(self) -> dict:
        """Serialize token usage for storage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_provider": dict(self.by_provider),
            "llm_calls": self.llm_calls,
            "fuzzy_routed": self.fuzzy_routed,
            "cache_hits": self.cache_hits,
            "total_judged": self.total_judged,
        }

    def reset(self):
        self.llm_calls = 0
        self.fuzzy_routed = 0
        self.cache_hits = 0
        self.total_judged = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.by_provider = {}


class LLMJudge:
    """
    LLM-as-Judge for evaluating MCP server / agent responses.

    Provider priority: Cerebras → Groq → OpenRouter → Fuzzy fallback.
    Supports key rotation: comma-separated keys in env vars rotate on rate limit.
    Results are cached to avoid duplicate API calls.
    """

    # Provider → base URL mapping (all OpenAI-compatible)
    _PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "openrouter": "https://openrouter.ai/api/v1",
        "mistral": "https://api.mistral.ai/v1",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        fallback_key: Optional[str] = None,
        fallback_model: str = "deepseek-chat",
        fallback_provider: str = "deepseek",
        fallback2_key: Optional[str] = None,
        fallback2_model: str = "llama-3.3-70b-versatile",
        fallback2_provider: str = "groq",
    ):
        # Support comma-separated keys for rotation
        self._primary_rotator = _KeyRotator(api_key) if api_key else None
        self.api_key = self._primary_rotator.current if self._primary_rotator else None
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self._fallback_rotator = _KeyRotator(fallback_key) if fallback_key else None
        self.fallback_key = self._fallback_rotator.current if self._fallback_rotator else None
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider
        self._fallback2_rotator = _KeyRotator(fallback2_key) if fallback2_key else None
        self.fallback2_key = self._fallback2_rotator.current if self._fallback2_rotator else None
        self.fallback2_model = fallback2_model
        self.fallback2_provider = fallback2_provider
        self._cache: Dict[str, CacheEntry] = {}
        self._llm_available = bool(self.api_key)
        self.metrics = JudgeMetrics()

        if self._llm_available:
            keys_info = []
            if self._primary_rotator:
                keys_info.append(f"{provider}:{self._primary_rotator.key_count} keys")
            if self._fallback_rotator:
                keys_info.append(f"{fallback_provider}:{self._fallback_rotator.key_count} keys")
            if self._fallback2_rotator:
                keys_info.append(f"{fallback2_provider}:{self._fallback2_rotator.key_count} keys")
            logger.info(f"LLM Judge: provider={provider}, model={model}, rotation=[{', '.join(keys_info)}]")
        else:
            logger.info("LLM Judge: using fuzzy fallback (no API key)")

    def reset_keys(self):
        """Reset exhausted API keys across all providers. Call between evaluations."""
        for rotator in [self._primary_rotator, self._fallback_rotator, self._fallback2_rotator]:
            if rotator:
                rotator.reset_exhausted()

    def log_metrics(self):
        """Log optimization metrics summary. Call at end of evaluation."""
        m = self.metrics
        if m.total_judged == 0:
            return
        logger.info(f"[Optimization] {m.summary()}")

    def _provider_base_url(self, provider: str) -> str:
        """Get default base URL for a provider."""
        return self._PROVIDER_URLS.get(provider, "https://api.openai.com/v1")

    @property
    def is_llm_available(self) -> bool:
        return self._llm_available

    def _cache_key(self, question: str, expected: str, answer: str) -> str:
        raw = f"{question}|{expected}|{answer}".lower().strip()
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[JudgeResult]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > CACHE_TTL:
            del self._cache[key]
            return None
        return JudgeResult(
            score=entry.result.score,
            explanation=entry.result.explanation,
            method=entry.result.method,
            cached=True,
        )

    def _store_cache(self, key: str, result: JudgeResult) -> None:
        self._cache[key] = CacheEntry(result=result, timestamp=time.time())
        if len(self._cache) > 1000:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

    def _build_prompt(self, question: str, expected: str, answer: str) -> str:
        return (
            "You are a strict judge evaluating an MCP tool response.\n\n"
            "Scoring rubric:\n"
            "  90-100: Correct result with all expected fields/values present\n"
            "  70-89:  Correct result but minor missing fields or formatting differences\n"
            "  50-69:  Partially correct — some expected data present, some missing or wrong\n"
            "  25-49:  Poor — response exists but mostly wrong or irrelevant\n"
            "  0-24:   Fail — empty, crash, or completely wrong response\n\n"
            "If the expected behavior describes an error case and the actual response "
            "IS an error/validation message, score 80-95 (the tool correctly rejected bad input).\n\n"
            f"Question/Input: {question}\n"
            f"Expected behavior: {expected}\n"
            f"Actual response: {answer}\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"score": <0-100>, "explanation": "<brief 1-sentence explanation>"}\n'
        )

    def _parse_response(self, text: str) -> Optional[Tuple[int, str]]:
        text = text.strip()
        if "```" in text:
            for segment in text.split("```"):
                segment = segment.strip()
                if segment.startswith("json"):
                    segment = segment[4:].strip()
                if segment.startswith("{"):
                    text = segment
                    break
        try:
            data = json.loads(text)
            score = max(0, min(100, int(data.get("score", 0))))
            explanation = str(data.get("explanation", ""))
            return score, explanation
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    async def _try_provider_with_rotation(
        self, question: str, expected: str, answer: str,
        rotator: _KeyRotator, model: str, provider: str, base_url: str,
    ) -> Optional[JudgeResult]:
        """Try a provider with key rotation on rate limit."""
        attempts = rotator.key_count
        for i in range(attempts):
            key = rotator.current
            if not key:
                break
            result = await self._call_llm(
                question, expected, answer,
                key, model, provider, base_url,
            )
            if result is not None:
                return result
            # _call_llm returned None — likely 429 or quota error, rotate key
            logger.info(f"Key rotation: {provider} key #{i+1}/{attempts} failed, rotating")
            rotator.rotate(exhausted=True)
        return None

    async def ajudge(self, question: str, expected: str, answer: str, test_type: str = "") -> JudgeResult:
        """Judge a response asynchronously. Primary → fallback → fuzzy with key rotation.

        If test_type is a simple type (error_handling, type_coercion, boundary),
        skips the LLM entirely and uses the fuzzy scorer directly — saves ~$0 per call.
        """
        self.metrics.total_judged += 1

        # QO-043: Sanitize MCP response before judging to block adversarial manipulation
        from src.core.judge_sanitizer import sanitize_judge_input
        san = sanitize_judge_input(answer)
        if san.had_detections:
            self.metrics.sanitized_inputs = getattr(self.metrics, "sanitized_inputs", 0) + 1
        answer = san.sanitized_text

        # Compute result (single-return refactor for QO-047 audit logging)
        result = await self._ajudge_inner(question, expected, answer, test_type)

        # QO-047: Audit log judge call + sanitization event (best-effort, non-blocking)
        try:
            from src.core.mcp_client import current_evaluation_id
            eval_id = current_evaluation_id.get()
            if eval_id and result is not None:
                from src.core.audit_log import log_judge_call, log_sanitization
                # Track call index per-evaluation via the LLMJudge instance
                self._audit_call_index = getattr(self, "_audit_call_index", 0) + 1
                await log_judge_call(
                    evaluation_id=eval_id,
                    call_index=self._audit_call_index,
                    provider=result.provider or self.provider,
                    model=getattr(result, "model", self.model),
                    question=question,
                    expected=expected,
                    answer=answer,
                    raw_response_text=getattr(result, "raw_response", ""),
                    parsed_score=result.score,
                    parsed_explanation=result.explanation,
                    method=result.method,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                )
                # Sanitization event (only logged if had_detections)
                await log_sanitization(
                    evaluation_id=eval_id,
                    judge_call_index=self._audit_call_index,
                    sanitization_result=san,
                )
        except Exception as e:
            logger.debug(f"Audit log skipped: {e}")

        return result

    async def _ajudge_inner(self, question: str, expected: str, answer: str, test_type: str) -> JudgeResult:
        """Inner judge logic — extracted for QO-047 single-return audit hooking."""
        # Optimization: route simple test types directly to fuzzy scorer
        if test_type in FUZZY_ROUTABLE_TEST_TYPES:
            start = time.time()
            result = self._judge_fuzzy(question, expected, answer)
            result.method = "fuzzy_routed"
            result.latency_ms = int((time.time() - start) * 1000)
            self.metrics.fuzzy_routed += 1
            return result

        key = self._cache_key(question, expected, answer)
        cached = self._get_cached(key)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached

        start = time.time()

        # Try primary provider (with key rotation)
        if self._primary_rotator and self._primary_rotator.current:
            result = await self._try_provider_with_rotation(
                question, expected, answer,
                self._primary_rotator, self.model, self.provider, self.base_url,
            )
            if result is not None:
                result.latency_ms = int((time.time() - start) * 1000)
                self._store_cache(key, result)
                self.metrics.llm_calls += 1
                self.metrics.record_tokens(result.provider or self.provider, result.input_tokens, result.output_tokens)
                return result

        # Try fallback provider (with key rotation)
        if self._fallback_rotator and self._fallback_rotator.current:
            result = await self._try_provider_with_rotation(
                question, expected, answer,
                self._fallback_rotator, self.fallback_model, self.fallback_provider,
                self._provider_base_url(self.fallback_provider),
            )
            if result is not None:
                result.latency_ms = int((time.time() - start) * 1000)
                self._store_cache(key, result)
                self.metrics.llm_calls += 1
                self.metrics.record_tokens(result.provider or self.fallback_provider, result.input_tokens, result.output_tokens)
                return result

        # Try fallback2 provider (with key rotation)
        if self._fallback2_rotator and self._fallback2_rotator.current:
            result = await self._try_provider_with_rotation(
                question, expected, answer,
                self._fallback2_rotator, self.fallback2_model, self.fallback2_provider,
                self._provider_base_url(self.fallback2_provider),
            )
            if result is not None:
                result.latency_ms = int((time.time() - start) * 1000)
                self._store_cache(key, result)
                self.metrics.llm_calls += 1
                self.metrics.record_tokens(result.provider or self.fallback2_provider, result.input_tokens, result.output_tokens)
                return result

        # Fuzzy fallback
        result = self._judge_fuzzy(question, expected, answer)
        result.latency_ms = int((time.time() - start) * 1000)
        self._store_cache(key, result)
        return result

    async def _call_llm(
        self, question: str, expected: str, answer: str,
        api_key: str, model: str, provider: str, base_url: str,
    ) -> Optional[JudgeResult]:
        prompt = self._build_prompt(question, expected, answer)
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise scoring judge. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 512 if provider == "cerebras" else 150,
        }

        # Rate limit per provider
        rate_limiter = _ProviderRateLimiter.for_provider(provider)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for attempt in range(MAX_RETRIES):
                    await rate_limiter.wait()
                    response = await client.post(url, headers=headers, json=body)
                    if response.status_code == 429:
                        # Check if it's a permanent quota error vs temporary rate limit
                        try:
                            err_body = response.json()
                            err_code = err_body.get("error", {}).get("code", "")
                            if err_code == "insufficient_quota":
                                logger.warning(f"{provider} quota exhausted, skipping retries")
                                return None
                        except Exception:
                            pass
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(f"Judge rate limited (429), retry {attempt + 1}/{MAX_RETRIES}")
                        await asyncio.sleep(delay)
                        continue
                    break
                else:
                    # All retries exhausted with 429 — mark rate limited and move on
                    if response.status_code == 429:
                        logger.warning(f"{provider} API returned 429 after {MAX_RETRIES} retries — skipping to next provider")
                        # Bump the rate limiter's last_call forward so next attempt waits,
                        # but don't block the current coroutine with a long sleep.
                        async with rate_limiter._lock:
                            rate_limiter._last_call = time.time() + 15
                        return None

                if response.status_code != 200:
                    logger.warning(f"{provider} API returned {response.status_code}")
                    return None

                data = response.json()
                message = data["choices"][0]["message"]
                # Some models (e.g. gpt-oss-120b) return reasoning instead of content
                text = message.get("content") or message.get("reasoning") or ""
                parsed = self._parse_response(text)
                if parsed is None:
                    return None

                # Extract token usage from provider response
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                score, explanation = parsed
                return JudgeResult(
                    score=score, explanation=explanation, method="llm",
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    provider=provider,
                )

        except Exception as e:
            logger.warning(f"LLM judge error ({provider}): {e}")
            return None

    def _judge_fuzzy(self, question: str, expected: str, answer: str) -> JudgeResult:
        """Format-aware fuzzy matching fallback.

        Routes to specialized scorers based on answer format:
        - JSON responses → _score_json_response()
        - Error strings  → _score_error_response()
        - Plain text     → _score_text_response()
        """
        if not answer or not answer.strip():
            return JudgeResult(score=0, explanation="Empty response", method="fuzzy")

        answer_type = _classify_answer(answer)

        if answer_type == "json":
            result = _score_json_response(expected, answer)
            if result is not None:
                score, explanation = result
                return JudgeResult(score=score, explanation=explanation, method="fuzzy")
            # JSON parse failed in scorer — fall through to text

        if answer_type == "error":
            score, explanation = _score_error_response(expected, answer)
            return JudgeResult(score=score, explanation=explanation, method="fuzzy")

        return self._score_text_response(expected, answer)

    def _score_text_response(self, expected: str, answer: str) -> JudgeResult:
        """Original fuzzy algorithm with stop-word filtering."""
        answer_lower = answer.lower().strip()
        expected_lower = expected.lower().strip()

        seq_ratio = SequenceMatcher(None, expected_lower, answer_lower).ratio()

        expected_terms = _filter_content_terms(expected)
        if expected_terms:
            answer_terms = answer_lower.split()
            term_scores = []
            for et in expected_terms:
                if et in answer_lower:
                    term_scores.append(1.0)
                elif answer_terms:
                    best = max(SequenceMatcher(None, et, at).ratio() for at in answer_terms)
                    term_scores.append(best)
                else:
                    term_scores.append(0.0)
            keyword_score = sum(term_scores) / len(term_scores)
        else:
            keyword_score = 0.0

        containment_score = 1.0 if expected_lower in answer_lower else (
            0.7 if answer_lower in expected_lower else 0.0
        )

        raw_score = keyword_score * 0.50 + seq_ratio * 0.30 + containment_score * 0.20
        score = max(0, min(100, int(round(raw_score * 100))))

        return JudgeResult(
            score=score,
            explanation=f"Fuzzy: keyword={keyword_score:.0%}, similarity={seq_ratio:.0%}",
            method="fuzzy",
        )
