# Quality Oracle — Code Patterns

> Reference patterns for implementing new features in quality-oracle.

---

## API Endpoint Pattern

All endpoints follow this structure in `src/api/v1/`:

```python
# src/api/v1/{feature}.py
from fastapi import APIRouter, Depends, HTTPException
from src.auth.dependencies import get_api_key
from src.storage.mongodb import {collection}_col
from src.storage.models import {Model}

router = APIRouter()

@router.get("/{feature}/{id}")
async def get_item(id: str, api_key: str = Depends(get_api_key)):
    doc = await {collection}_col().find_one({"_id": id})
    if not doc:
        raise HTTPException(404, "Not found")
    return {Model}(**doc)

@router.post("/{feature}")
async def create_item(req: {RequestModel}, api_key: str = Depends(get_api_key)):
    # ... business logic
    await {collection}_col().insert_one(doc)
    return {ResponseModel}(**doc)
```

**Register in `src/main.py`:**
```python
from src.api.v1.{feature} import router as {feature}_router
app.include_router({feature}_router, prefix="/v1", tags=["{feature}"])
```

---

## Public Endpoint Pattern (No Auth)

For endpoints like `/v1/scan`, `/v1/health`, `/v1/badge`:

```python
@router.post("/scan")
async def scan(req: ScanRequest):  # No Depends(get_api_key)
    # IP-based rate limiting instead
    await check_ip_rate_limit(request)
    # ... logic
```

---

## MongoDB Collection Pattern

```python
# src/storage/mongodb.py — add accessor function
def new_collection_col():
    return _db()["quality__new_collection"]

# Usage anywhere:
from src.storage.mongodb import new_collection_col
col = new_collection_col()
await col.find_one({"key": value})
await col.insert_one(doc)
await col.update_one({"_id": id}, {"$set": updates})
```

**Naming:** All collections prefixed `quality__` (e.g., `quality__evaluations`, `quality__scores`).

---

## Pydantic Model Pattern

```python
# src/storage/models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class FeatureRequest(BaseModel):
    url: str
    level: str = "verified"

class FeatureResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    result: Optional[dict] = None
```

---

## Background Task Pattern

For long-running operations (evaluations, batch scoring):

```python
@router.post("/evaluate")
async def start_eval(req: EvaluateRequest, bg: BackgroundTasks, api_key: str = Depends(get_api_key)):
    eval_id = str(uuid4())
    await evaluations_col().insert_one({"_id": eval_id, "status": "pending"})
    bg.add_task(_run_evaluation, eval_id, req)
    return {"eval_id": eval_id, "status": "pending"}

async def _run_evaluation(eval_id: str, req: EvaluateRequest):
    try:
        await evaluations_col().update_one({"_id": eval_id}, {"$set": {"status": "running"}})
        result = await evaluator.evaluate_full(...)
        await evaluations_col().update_one({"_id": eval_id}, {"$set": {"status": "completed", "result": result}})
    except Exception as e:
        await evaluations_col().update_one({"_id": eval_id}, {"$set": {"status": "failed", "error": str(e)}})
```

---

## Core Module Pattern

New evaluation/analysis modules go in `src/core/`:

```python
# src/core/new_feature.py
"""
One-line description of what this module does.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class FeatureResult:
    score: float
    details: List[str] = field(default_factory=list)

async def run_feature(tools: list, **kwargs) -> FeatureResult:
    """Main entry point for the feature."""
    # ... implementation
    return FeatureResult(score=0.0)
```

---

## Test Pattern

```python
# tests/test_{feature}.py
import pytest
from src.core.{feature} import {function}

class TestFeatureName:
    """Group related tests."""

    def test_basic_case(self):
        result = {function}(input)
        assert result.score > 0

    def test_edge_case(self):
        result = {function}(edge_input)
        assert result.score == 0

    @pytest.mark.asyncio
    async def test_async_operation(self):
        result = await {async_function}(input)
        assert result is not None
```

**Run tests:** `python3 -m pytest tests/ -v`
**Run single file:** `python3 -m pytest tests/test_{feature}.py -v`
**Lint:** `ruff check src/`

---

## Redis Cache Pattern

```python
from src.storage.redis_client import get_redis

async def get_cached_or_compute(key: str, compute_fn, ttl: int = 3600):
    redis = await get_redis()
    cached = await redis.get(f"qo:{key}")
    if cached:
        return json.loads(cached)
    result = await compute_fn()
    await redis.set(f"qo:{key}", json.dumps(result), ex=ttl)
    return result
```

**Prefix:** All Redis keys use `qo:` prefix.

---

## Config Pattern

```python
# src/config.py — add new settings
class Settings(BaseSettings):
    new_feature_enabled: bool = False
    new_feature_threshold: float = 0.5

# Usage:
from src.config import settings
if settings.new_feature_enabled:
    ...
```

---

## LLM Judge Integration

When a feature needs LLM-based scoring:

```python
from src.core.llm_judge import LLMJudge

judge = LLMJudge()
response = await judge.evaluate(
    prompt="Score this response for accuracy...",
    response_text=agent_response,
)
# Returns: {"score": 75, "explanation": "..."}
```

For consensus (2-3 judges):
```python
from src.core.consensus_judge import ConsensusJudge

consensus = ConsensusJudge()
result = await consensus.evaluate(prompt, response_text)
# Returns median/majority score from 2-3 independent judges
```

---

## Static Analysis Pattern (No LLM)

For checks that don't need LLM (like tool poisoning, schema validation):

```python
import re
from dataclasses import dataclass

_PATTERNS = [
    re.compile(r"suspicious pattern", re.IGNORECASE),
]

@dataclass
class CheckResult:
    detected: bool
    findings: list[str]

def check_static(tools: list) -> CheckResult:
    findings = []
    for tool in tools:
        text = tool.get("description", "")
        for pattern in _PATTERNS:
            if match := pattern.search(text):
                findings.append(f"Tool '{tool.get('name')}': {match.group()[:60]}")
    return CheckResult(detected=len(findings) > 0, findings=findings)
```
