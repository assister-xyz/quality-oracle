"""A2A Agent Card — AgentTrust as a discoverable A2A agent (v0.3 spec)."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.standards.a2a_extension import build_provider_extension_declaration

router = APIRouter()


@router.get("/.well-known/agent.json")
async def agent_card():
    """
    A2A v0.3-compliant Agent Card for AgentTrust.

    AgentTrust IS an A2A agent that can be discovered and interacted with
    by other agents using the Google A2A protocol.
    """
    return {
        "name": "AgentTrust",
        "description": "Active competency verification for AI agents, MCP servers, and skills. "
                       "Submit any agent or MCP server for quality evaluation and receive "
                       "a verifiable quality score with AQVC (W3C VC) attestation.",
        "url": "https://laureum.ai",
        "version": "0.1.0",
        "protocolVersion": "0.3.0",
        "provider": {
            "organization": "Assisterr",
            "url": "https://assisterr.ai",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extensions": [build_provider_extension_declaration()],
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "evaluate-quality",
                "name": "Evaluate Agent/MCP Server Quality",
                "description": "Submit an MCP server URL or agent endpoint for quality evaluation. "
                               "Returns a quality score (0-100), tier (expert/proficient/basic/failed), "
                               "and AQVC (W3C Verifiable Credential) attestation.",
                "tags": ["quality", "evaluation", "verification", "mcp", "agent"],
                "examples": [
                    "Evaluate the quality of mcp-server-github",
                    "Check quality score for https://example.com/mcp-server",
                    "Get quality attestation for agent-xyz",
                ],
            },
            {
                "id": "lookup-score",
                "name": "Lookup Quality Score",
                "description": "Look up the existing quality score for a previously evaluated target.",
                "tags": ["score", "lookup", "quality"],
                "examples": [
                    "What is the quality score for mcp-server-github?",
                    "Find the best MCP servers for code generation",
                ],
            },
            {
                "id": "verify-attestation",
                "name": "Verify Quality Attestation",
                "description": "Verify the validity and authenticity of a quality attestation credential.",
                "tags": ["verify", "attestation", "credential"],
                "examples": [
                    "Verify attestation urn:uuid:abc-123",
                    "Is this quality credential still valid?",
                ],
            },
        ],
    }


@router.get("/.well-known/did.json")
async def did_document():
    """DID Document endpoint for W3C VC verification.

    Returns the DID Document with Multikey verification method,
    allowing anyone to verify AgentTrust's VCs offline.
    """
    from src.core.attestation import _get_or_generate_key
    from src.standards.vc_issuer import build_did_document
    from src.config import settings

    key = _get_or_generate_key()
    return build_did_document(key.public_key(), issuer_did=settings.jwt_issuer)


@router.get("/contexts/quality/v1")
async def quality_context():
    """Custom JSON-LD context defining AgentQualityCredential vocabulary.

    Provides term definitions for the AgentTrust VC credentialSubject fields.
    """
    return JSONResponse(content={
        "@context": {
            "@version": 1.1,
            "at": "https://laureum.ai/vocab#",
            "AgentQualityCredential": "at:AgentQualityCredential",
            "QualityEvaluation": "at:QualityEvaluation",
            "qualityScore": {"@id": "at:qualityScore", "@type": "https://www.w3.org/2001/XMLSchema#integer"},
            "qualityTier": {"@id": "at:qualityTier", "@type": "https://www.w3.org/2001/XMLSchema#string"},
            "confidence": {"@id": "at:confidence", "@type": "https://www.w3.org/2001/XMLSchema#float"},
            "evaluationLevel": {"@id": "at:evaluationLevel", "@type": "https://www.w3.org/2001/XMLSchema#integer"},
            "domains": {"@id": "at:domains", "@container": "@set"},
            "questionsAsked": {"@id": "at:questionsAsked", "@type": "https://www.w3.org/2001/XMLSchema#integer"},
            "evaluationId": {"@id": "at:evaluationId", "@type": "https://www.w3.org/2001/XMLSchema#string"},
            "evaluatedAt": {"@id": "at:evaluatedAt", "@type": "https://www.w3.org/2001/XMLSchema#dateTime"},
        },
    }, media_type="application/ld+json")


@router.get("/ext/evaluation/v1")
async def extension_spec():
    """Extension specification document for AgentTrust evaluation.

    Returns the JSON-LD-style schema describing the extension's roles and parameters.
    """
    return {
        "@context": "https://laureum.ai/ext/evaluation/v1",
        "name": "AgentTrust Evaluation Extension",
        "version": "1.0",
        "roles": ["provider", "verified_subject"],
        "params_schema": {
            "provider": {
                "evaluation_levels": "array",
                "supported_targets": "array",
                "attestation_format": "string",
            },
            "verified_subject": {
                "score": "integer(0-100)",
                "tier": "string",
                "confidence": "float",
                "last_evaluated": "string (ISO 8601)",
                "attestation_url": "string",
                "badge_url": "string",
                "verify_url": "string",
            },
        },
    }
