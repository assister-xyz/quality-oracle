# Framework Integration Specs

## Overview

AgentTrust can integrate with AI agent frameworks as a **pre-delegation quality gate**: before an orchestrator delegates a task to a sub-agent or MCP server, it verifies competency via AgentTrust.

All integrations follow the same pattern:
1. Call AgentTrust API (`POST /v1/evaluate` or `GET /v1/score/{id}`)
2. Check score against threshold
3. Proceed or reject based on quality

---

## 1. LangChain Community Integration

**Target:** PR to [langchain-ai/langchain-community](https://github.com/langchain-ai/langchain-community)

**File:** `langchain_community/tools/agenttrust/tool.py`

**Implementation:**
```python
"""AgentTrust quality verification tool for LangChain."""
from langchain_core.tools import BaseTool
from pydantic import Field
import httpx


class AgentTrustEvaluator(BaseTool):
    """Evaluate an MCP server or AI agent's quality before delegation.

    Returns a quality score (0-100), tier, and confidence level.
    Use this before delegating tasks to verify competency.
    """
    name: str = "agenttrust_evaluate"
    description: str = (
        "Evaluate the quality of an MCP server or AI agent. "
        "Input: server URL. Output: score, tier, confidence."
    )
    api_url: str = Field(default="http://localhost:8002")
    api_key: str = Field(default="")
    min_score: int = Field(default=60, description="Minimum acceptable score")

    def _run(self, server_url: str) -> str:
        """Synchronous evaluation."""
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = httpx.post(
            f"{self.api_url}/v1/evaluate",
            json={"target_url": server_url, "level": 2},
            headers=headers,
            timeout=120.0,
        )
        data = resp.json()
        score = data.get("score", 0)
        tier = data.get("tier", "unknown")

        if score >= self.min_score:
            return f"PASSED: score={score}, tier={tier}"
        return f"FAILED: score={score} < {self.min_score}, tier={tier}"

    async def _arun(self, server_url: str) -> str:
        """Async evaluation."""
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.api_url}/v1/evaluate",
                json={"target_url": server_url, "level": 2},
                headers=headers,
            )
        data = resp.json()
        score = data.get("score", 0)
        tier = data.get("tier", "unknown")

        if score >= self.min_score:
            return f"PASSED: score={score}, tier={tier}"
        return f"FAILED: score={score} < {self.min_score}, tier={tier}"
```

**Status:** [ ] Code ready, [ ] Tests written, [ ] PR submitted

---

## 2. CrewAI Tool

**Target:** PR to [crewAIInc/crewAI-tools](https://github.com/crewAIInc/crewAI-tools) or standalone package

**Implementation:**
```python
"""AgentTrust quality gate for CrewAI workflows."""
from crewai.tools import BaseTool
import httpx


class AgentTrustTool(BaseTool):
    name: str = "AgentTrust Quality Check"
    description: str = (
        "Verify an MCP server or AI agent's quality before delegating tasks. "
        "Returns quality score (0-100), tier, and pass/fail status."
    )

    def _run(self, server_url: str) -> str:
        api_url = "http://localhost:8002"
        resp = httpx.post(
            f"{api_url}/v1/evaluate",
            json={"target_url": server_url, "level": 2},
            timeout=120.0,
        )
        data = resp.json()
        return f"Score: {data.get('score', 0)}/100, Tier: {data.get('tier', 'unknown')}"
```

**Status:** [ ] Code ready, [ ] Submitted to awesome-crewai

---

## 3. ElizaOS Plugin

**Target:** npm package `plugin-agenttrust` using [plugin-starter](https://github.com/elizaOS/eliza-plugin-starter)

**Implementation sketch (TypeScript):**
```typescript
// plugin-agenttrust/src/index.ts
import { Plugin, Action } from "@elizaos/core";

const evaluateAction: Action = {
  name: "EVALUATE_AGENT_QUALITY",
  description: "Evaluate an AI agent or MCP server quality via AgentTrust",
  handler: async (runtime, message) => {
    const serverUrl = message.content.text;
    const apiUrl = runtime.getSetting("AGENTTRUST_API_URL") || "http://localhost:8002";

    const resp = await fetch(`${apiUrl}/v1/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_url: serverUrl, level: 2 }),
    });
    const data = await resp.json();

    return {
      text: `Quality Score: ${data.score}/100 (${data.tier})\nConfidence: ${data.confidence}`,
    };
  },
};

export const agentTrustPlugin: Plugin = {
  name: "agenttrust",
  description: "Quality verification for AI agents via AgentTrust",
  actions: [evaluateAction],
};
```

**Status:** [ ] Skeleton created, [ ] Published to npm

---

## 4. promptfoo Grader Plugin

**Target:** Custom grader for [promptfoo](https://github.com/promptfoo/promptfoo)

**Implementation:**
```yaml
# promptfoo config: promptfooconfig.yaml
providers:
  - id: mcp:agenttrust
    config:
      serverUrl: "http://localhost:8003/sse"

tests:
  - vars:
      server_url: "http://target-server:8010/sse"
    assert:
      - type: javascript
        value: "output.score >= 70"
      - type: javascript
        value: "output.tier !== 'failing'"
```

**Status:** [ ] Config template ready

---

## 5. NVIDIA garak Probe Contribution

**Target:** PR to [NVIDIA/garak](https://github.com/NVIDIA/garak)

**Probes to contribute (from our adversarial.py):**
1. `AgentPromptInjection` — Tests agent resistance to instruction injection
2. `AgentPIILeakage` — Tests if agent leaks PII from context
3. `AgentHallucination` — Tests factual grounding of agent responses
4. `AgentOverflow` — Tests input boundary handling
5. `AgentSystemPromptExtraction` — Tests system prompt protection

**Status:** [ ] Probes adapted to garak format, [ ] PR submitted

---

## Priority Order

1. **LangChain** — Highest stars (129K), explicit "open to contributions" policy
2. **CrewAI** — Growing fast (45.9K), tool ecosystem is young
3. **ElizaOS** — Web3/Solana native, $20B+ agent market cap
4. **promptfoo** — Eval ecosystem, natural fit
5. **garak** — Security angle, NVIDIA backing gives credibility
