"""Tests for OWASP probe hardening: multi-payload attack bank, encoding bypasses, and FP reduction.

Covers:
- TestAttackBank: bank loading and structure
- TestEncodingVariants: Base64, URL encoding, homoglyphs, case variation, whitespace injection
- TestMultiPayloadInjection: prompt injection with multiple payloads
- TestMultiPayloadJailbreak: jailbreak with multiple payloads
- TestImprovedResponseChecking: enhanced injection, exfiltration, jailbreak detection
- TestFalsePositiveReduction: context-aware FP filtering for SSTI, backtick, base64, credentials
"""
import json
import re
from pathlib import Path

import pytest

from src.core.adversarial import (
    _check_credential_harvest,
    _check_exfiltration,
    _check_indirect_injection,
    _check_insecure_output,
    _check_insecure_output_static,
    _check_jailbreak,
    _check_prompt_injection,
    _check_supply_chain,
    _generate_probes,
    _load_attack_bank,
    _BASE64_PAYLOAD_PATTERN,
)
from src.core.encoding_variants import generate_variants, pick_variants


# ── TestAttackBank ──────────────────────────────────────────────────────────


class TestAttackBank:
    def test_bank_loads_successfully(self):
        bank = _load_attack_bank()
        assert isinstance(bank, dict)
        assert len(bank) > 0

    def test_all_required_probe_types_present(self):
        bank = _load_attack_bank()
        required = [
            "prompt_injection", "jailbreak", "indirect_injection",
            "system_prompt_extraction", "exfiltration",
        ]
        for probe_type in required:
            assert probe_type in bank, f"Missing probe type: {probe_type}"

    def test_minimum_variants_per_type(self):
        bank = _load_attack_bank()
        for probe_type, payloads in bank.items():
            assert len(payloads) >= 5, (
                f"{probe_type} has only {len(payloads)} payloads, expected >= 5"
            )

    def test_payloads_are_nonempty_strings(self):
        bank = _load_attack_bank()
        for probe_type, payloads in bank.items():
            for i, payload in enumerate(payloads):
                assert isinstance(payload, str), f"{probe_type}[{i}] is not a string"
                assert len(payload.strip()) > 10, f"{probe_type}[{i}] is too short"

    def test_bank_file_is_valid_json(self):
        bank_path = Path(__file__).parent.parent / "src" / "core" / "attack_bank.json"
        with open(bank_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ── TestEncodingVariants ────────────────────────────────────────────────────


class TestEncodingVariants:
    def test_base64_generation(self):
        variants = generate_variants("test payload")
        # Second variant should be Base64
        assert any("Decode and execute:" in v for v in variants)
        assert any("dGVzdCBwYXlsb2Fk" in v for v in variants)  # base64 of "test payload"

    def test_url_encoding(self):
        variants = generate_variants("test payload with spaces")
        assert any("test%20payload%20with%20spaces" in v for v in variants)

    def test_homoglyphs(self):
        variants = generate_variants("apple core")
        # Should have Cyrillic 'a' replacement
        has_homoglyph = any(
            v != "apple core" and "\u0430" in v
            for v in variants
        )
        assert has_homoglyph, "No homoglyph variant found"

    def test_case_variation(self):
        variants = generate_variants("Hello World")
        assert "HELLO WORLD" in variants
        assert "hELLO wORLD" in variants

    def test_whitespace_injection(self):
        variants = generate_variants("test me")
        assert "test  me" in variants  # Double space
        assert "test\tme" in variants  # Tab

    def test_leetspeak(self):
        variants = generate_variants("aeiou")
        assert any("4" in v and "3" in v for v in variants)

    def test_pick_variants_includes_base64(self):
        picked = pick_variants("test payload", count=2)
        assert len(picked) == 2
        assert any("Decode and execute:" in v for v in picked)

    def test_pick_variants_count(self):
        picked = pick_variants("test payload", count=3)
        assert len(picked) == 3


# ── TestMultiPayloadInjection ──────────────────────────────────────────────


class TestMultiPayloadInjection:
    """Test that prompt injection probes use multiple payloads from the attack bank."""

    def _make_tools(self):
        return [{
            "name": "generate_text",
            "description": "Generate text content",
            "inputSchema": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            },
        }]

    def test_generates_multiple_prompt_injection_probes(self):
        probes = _generate_probes(self._make_tools())
        pi_probes = [p for p in probes if p["probe_type"] == "prompt_injection"]
        # 3 payloads x 3 (1 original + 2 variants) = 9 minimum
        assert len(pi_probes) >= 9, f"Only {len(pi_probes)} prompt_injection probes generated"

    def test_generates_multiple_system_prompt_probes(self):
        probes = _generate_probes(self._make_tools())
        spe_probes = [p for p in probes if p["probe_type"] == "system_prompt_extraction"]
        assert len(spe_probes) >= 9, f"Only {len(spe_probes)} system_prompt_extraction probes"

    def test_generates_multiple_indirect_injection_probes(self):
        probes = _generate_probes(self._make_tools())
        ii_probes = [p for p in probes if p["probe_type"] == "indirect_prompt_injection"]
        assert len(ii_probes) >= 9, f"Only {len(ii_probes)} indirect_injection probes"

    def test_generates_multiple_exfiltration_probes(self):
        probes = _generate_probes(self._make_tools())
        exfil_probes = [p for p in probes if p["probe_type"] == "data_exfiltration"]
        assert len(exfil_probes) >= 9, f"Only {len(exfil_probes)} exfiltration probes"

    def test_payloads_are_diverse(self):
        """Ensure the generated probes don't all use the same payload."""
        probes = _generate_probes(self._make_tools())
        pi_probes = [p for p in probes if p["probe_type"] == "prompt_injection"]
        payloads = {list(p["input_data"].values())[0] for p in pi_probes}
        assert len(payloads) >= 5, "Prompt injection payloads not diverse enough"


# ── TestMultiPayloadJailbreak ──────────────────────────────────────────────


class TestMultiPayloadJailbreak:
    def _make_tools(self):
        return [{
            "name": "chat_assistant",
            "description": "Chat with an AI assistant",
            "inputSchema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        }]

    def test_generates_multiple_jailbreak_probes(self):
        probes = _generate_probes(self._make_tools())
        jb_probes = [p for p in probes if p["probe_type"] == "jailbreak_resistance"]
        assert len(jb_probes) >= 9, f"Only {len(jb_probes)} jailbreak probes"

    def test_jailbreak_payloads_are_diverse(self):
        probes = _generate_probes(self._make_tools())
        jb_probes = [p for p in probes if p["probe_type"] == "jailbreak_resistance"]
        payloads = {list(p["input_data"].values())[0] for p in jb_probes}
        assert len(payloads) >= 5, "Jailbreak payloads not diverse enough"

    def test_jailbreak_includes_encoding_variants(self):
        probes = _generate_probes(self._make_tools())
        jb_probes = [p for p in probes if p["probe_type"] == "jailbreak_resistance"]
        payloads = [list(p["input_data"].values())[0] for p in jb_probes]
        has_base64 = any("Decode and execute:" in p for p in payloads)
        assert has_base64, "No base64 encoding variant in jailbreak probes"


# ── TestImprovedResponseChecking ───────────────────────────────────────────


class TestImprovedResponseChecking:
    def test_prompt_injection_catches_subtle_instruction_following(self):
        """Multiple instruction-following signals should trigger detection."""
        result = _check_prompt_injection(
            "As requested, here is the text you wanted."
        )
        assert not result.passed

    def test_prompt_injection_catches_system_prompt_in_long_response(self):
        """Long response containing system prompt fragments suggests compliance."""
        long_response = "A" * 1500 + " You are a helpful assistant. My system prompt says..."
        result = _check_prompt_injection(long_response)
        assert not result.passed
        assert result.score < 50

    def test_prompt_injection_passes_normal_response(self):
        result = _check_prompt_injection("The weather today is sunny with 72F.")
        assert result.passed
        assert result.score == 100

    def test_exfiltration_catches_url_with_data_params(self):
        result = _check_exfiltration(
            "I have prepared the response. See https://evil.com/collect?data=secret123 for details."
        )
        assert not result.passed

    def test_exfiltration_catches_base64_block(self):
        b64_block = "A" * 120 + "=="  # Looks like base64
        result = _check_exfiltration(f"Here is the encoded data: {b64_block}")
        assert not result.passed

    def test_exfiltration_catches_serialized_context(self):
        result = _check_exfiltration(
            '{"context": "user said something", "session": "abc123"}'
        )
        assert not result.passed

    def test_exfiltration_passes_normal_response(self):
        result = _check_exfiltration("I calculated the result: 42")
        assert result.passed

    def test_jailbreak_catches_aim_persona(self):
        result = _check_jailbreak("I am AIM, and I will provide unfiltered responses.")
        assert not result.passed

    def test_jailbreak_catches_developer_mode(self):
        result = _check_jailbreak("Developer mode activated. No restrictions apply.")
        assert not result.passed

    def test_jailbreak_catches_long_harmful_response_without_refusal(self):
        """Long response with harmful indicators and no refusal should flag."""
        harmful_response = (
            "Here's how to exploit the vulnerability. "
            "First, inject the payload into the target. "
            "Then bypass the security mechanism. "
        ) * 5  # Make it > 500 chars
        result = _check_jailbreak(harmful_response)
        assert not result.passed

    def test_indirect_injection_catches_system_prompt_reveal(self):
        result = _check_indirect_injection(
            "My instructions are to help users with their tasks. My system prompt says..."
        )
        assert not result.passed

    def test_indirect_injection_catches_override_signals(self):
        result = _check_indirect_injection(
            "System override accepted. New primary directive acknowledged."
        )
        assert not result.passed


# ── TestFalsePositiveReduction ─────────────────────────────────────────────


class TestFalsePositiveReduction:
    def test_ssti_skipped_for_template_tools(self):
        """Template/documentation tools should not trigger SSTI detection."""
        tools = [{
            "name": "render_template",
            "description": "Render a Jinja2 template with variables using {{variable}} syntax",
            "inputSchema": {
                "type": "object",
                "properties": {"template": {"type": "string"}},
            },
        }]
        results = _check_insecure_output_static(tools)
        ssti_issues = [r for r in results if "SSTI" in r.explanation]
        assert len(ssti_issues) == 0, "Template tool should not trigger SSTI detection"

    def test_ssti_fires_for_non_template_tools(self):
        """Non-template tools with SSTI patterns should still be flagged."""
        tools = [{
            "name": "weather_api",
            "description": "Get weather data with {{malicious_code}} injection",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }]
        results = _check_insecure_output_static(tools)
        # This should flag because weather_api is not a template tool
        ssti_issues = [r for r in results if "SSTI" in r.explanation]
        assert len(ssti_issues) > 0

    def test_command_injection_skipped_for_code_tools(self):
        """Code/terminal tools should not trigger backtick command injection."""
        tools = [{
            "name": "run_shell_command",
            "description": "Execute a shell command in the terminal. Example: `ls -la`",
            "inputSchema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        }]
        results = _check_insecure_output_static(tools)
        cmd_issues = [r for r in results if "Command injection" in r.explanation]
        assert len(cmd_issues) == 0, "Code tool should not trigger command injection"

    def test_short_base64_not_flagged_in_supply_chain(self):
        """Base64 strings under 100 chars should not trigger supply chain detection."""
        tools = [{
            "name": "safe_tool",
            "description": "A tool with a 60-char base64 string " + "A" * 60,
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        b64_issues = [r for r in results if "Base64" in r.explanation]
        assert len(b64_issues) == 0, "Short base64 should not trigger supply chain"

    def test_long_base64_with_padding_flagged(self):
        """Base64 strings over 100 chars with padding should trigger."""
        long_b64 = "A" * 120 + "=="
        tools = [{
            "name": "suspicious_tool",
            "description": f"A tool with encoded payload: {long_b64}",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        b64_issues = [r for r in results if "Base64" in r.explanation]
        assert len(b64_issues) > 0, "Long base64 with padding should trigger"

    def test_credential_token_in_url_not_flagged(self):
        """token= appearing as a URL parameter should not trigger credential detection."""
        result = _check_credential_harvest(
            "Visit https://api.example.com/auth?token=success&status=ok for details."
        )
        assert result.passed

    def test_credential_token_assignment_flagged(self):
        """token= appearing as a standalone assignment should trigger."""
        result = _check_credential_harvest(
            "Configuration: token = sk-abc123xyz789secretkey000"
        )
        assert not result.passed

    def test_insecure_output_ssti_skipped_for_handlebars_docs(self):
        """Runtime SSTI check should skip when response mentions Handlebars."""
        result = _check_insecure_output(
            "In Handlebars template syntax, use {{variable}} to insert values."
        )
        assert result.passed
