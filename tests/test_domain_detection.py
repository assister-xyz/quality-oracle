"""Tests for domain auto-detection (QO-027)."""
from src.core.domain_detection import (
    detect_domain,
    detect_domain_with_confidence,
    detect_all_domains,
    get_domain_weights,
    DOMAIN_WEIGHTS,
    DOMAIN_CONFIDENCE_THRESHOLD,
)


class TestDetectDomain:
    def test_empty_tools(self):
        assert detect_domain([]) == "general"

    def test_search_tool(self):
        tools = [{"name": "web_search", "description": "Search the web for information"}]
        assert detect_domain(tools) == "search"

    def test_developer_tool(self):
        tools = [{"name": "git_commit", "description": "Commit changes to a git repository"}]
        assert detect_domain(tools) == "developer_tools"

    def test_finance_tool(self):
        tools = [{"name": "swap_tokens", "description": "Swap tokens on a DeFi exchange"}]
        assert detect_domain(tools) == "finance"

    def test_security_tool(self):
        tools = [{"name": "scan_vulnerabilities", "description": "Audit security vulnerabilities"}]
        assert detect_domain(tools) == "security"

    def test_data_tool(self):
        tools = [{"name": "query_db", "description": "Run SQL queries on PostgreSQL database"}]
        assert detect_domain(tools) == "data"

    def test_communication_tool(self):
        tools = [{"name": "send_email", "description": "Send email notification via Slack webhook"}]
        assert detect_domain(tools) == "communication"

    def test_content_tool(self):
        tools = [{"name": "summarize", "description": "Generate a summary of a document"}]
        assert detect_domain(tools) == "content"

    def test_no_keywords_returns_general(self):
        tools = [{"name": "foo", "description": "Does something"}]
        assert detect_domain(tools) == "general"

    def test_multiple_tools_strongest_wins(self):
        tools = [
            {"name": "search", "description": "Search for documents"},
            {"name": "search_web", "description": "Browse and find results online"},
            {"name": "git_status", "description": "Check git repository status"},
        ]
        # search has more keyword matches (search, find, browse)
        result = detect_domain(tools)
        assert result == "search"

    def test_description_and_name_both_checked(self):
        tools = [{"name": "tool1", "description": "Manage kubernetes deployments and docker containers for CI/CD build pipelines"}]
        assert detect_domain(tools) == "developer_tools"


class TestDetectAllDomains:
    def test_single_domain(self):
        tools = [{"name": "search_web", "description": "Search and find documents online"}]
        domains = detect_all_domains(tools, threshold=2)
        assert "search" in domains

    def test_multi_domain(self):
        tools = [
            {"name": "search_code", "description": "Search code in git repositories"},
            {"name": "lint_code", "description": "Lint and test code for bugs"},
        ]
        domains = detect_all_domains(tools, threshold=2)
        assert "developer_tools" in domains

    def test_empty_returns_general(self):
        assert detect_all_domains([]) == ["general"]

    def test_no_threshold_match_returns_general(self):
        tools = [{"name": "foo", "description": "bar"}]
        assert detect_all_domains(tools, threshold=2) == ["general"]

    def test_sorted_by_relevance(self):
        tools = [
            {"name": "search_vuln", "description": "Search and scan for security vulnerabilities, audit authentication"},
        ]
        domains = detect_all_domains(tools, threshold=2)
        # Security should be first (more keyword matches)
        assert domains[0] == "security"


class TestDetectDomainWithConfidence:
    def test_strong_match_high_confidence(self):
        tools = [{"name": "swap_tokens", "description": "Swap DeFi tokens on Solana exchange with best price"}]
        domain, conf = detect_domain_with_confidence(tools)
        assert domain == "finance"
        assert conf >= 0.5

    def test_weak_match_lower_confidence(self):
        # Single generic keyword match → moderate confidence at best
        tools = [{"name": "foo", "description": "Does something"}]
        domain, conf = detect_domain_with_confidence(tools)
        assert conf == 0.0  # no keywords match at all

    def test_empty_tools_zero_confidence(self):
        domain, conf = detect_domain_with_confidence([])
        assert domain == "general"
        assert conf == 0.0

    def test_ambiguous_match_lower_confidence(self):
        # "search" + "data" keywords both present
        tools = [{"name": "search_database", "description": "Search and query data from SQL database"}]
        domain, conf = detect_domain_with_confidence(tools)
        # Should have lower confidence due to ambiguity
        assert conf < 0.8

    def test_confidence_threshold_exists(self):
        assert 0.0 < DOMAIN_CONFIDENCE_THRESHOLD < 1.0


class TestGetDomainWeights:
    def test_general_weights(self):
        w = get_domain_weights("general")
        assert w["accuracy"] == 0.35
        assert abs(sum(w.values()) - 1.0) < 0.001

    def test_security_emphasizes_safety(self):
        w = get_domain_weights("security")
        assert w["safety"] > w["accuracy"]  # safety is emphasized
        assert w["safety"] == 0.35

    def test_search_emphasizes_latency(self):
        w = get_domain_weights("search")
        assert w["latency"] == 0.20  # higher than general's 0.10

    def test_unknown_domain_returns_general(self):
        w = get_domain_weights("nonexistent")
        assert w == DOMAIN_WEIGHTS["general"]

    def test_all_weights_sum_to_one(self):
        for domain, weights in DOMAIN_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"{domain} weights sum to {total}"
