# mcp-agenttrust

MCP server for AI agent and MCP server quality verification.

Evaluate any MCP server's competency before trusting it with real tasks or payments. Uses challenge-response testing across 6 quality dimensions with LLM judge consensus.

## Install

```bash
pip install mcp-agenttrust
```

## Usage

### With Claude Desktop / Cursor / Windsurf

Add to your MCP client config:

```json
{
  "mcpServers": {
    "agenttrust": {
      "command": "mcp-agenttrust",
      "env": {
        "AGENTTRUST_API_URL": "http://localhost:8002",
        "AGENTTRUST_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Standalone

```bash
export AGENTTRUST_API_URL=http://localhost:8002
mcp-agenttrust
```

## Tools

| Tool | Description |
|------|-------------|
| `check_quality(server_url)` | Full evaluation with LLM judge scoring |
| `get_score(target_id)` | Lookup cached quality score |
| `verify_attestation(attestation_id)` | Verify AQVC quality credential |
| `list_scores(domain, min_score)` | Search evaluated servers |

## Requirements

Requires a running [AgentTrust](https://github.com/vitaliiserbynassisterr/quality-oracle) API instance (the MCP server proxies requests to it).

## License

MIT
