"""Operator identity endpoints (QO-044 Sybil Defense).

POST /v1/operators/register — Register a new operator
GET  /v1/operators/{operator_id} — Get operator details
POST /v1/operators/{operator_id}/agents — Add an agent to an operator
GET  /v1/operators/agent/{target_id} — Find which operator owns an agent
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.auth.dependencies import get_api_key
from src.core.operator_identity import (
    register_operator,
    get_operator_by_id,
    get_operator_for_agent,
    add_agent_to_operator,
    DuplicateOperatorError,
    OperatorNotFoundError,
    MaxAgentsExceededError,
    OperatorError,
)
from src.storage.models import (
    OperatorRegisterRequest,
    OperatorRegisterResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class AddAgentRequest(BaseModel):
    agent_url: str


@router.post("/operators/register", response_model=OperatorRegisterResponse)
async def operators_register(
    request: OperatorRegisterRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """Register a new operator with email-based identity.

    Operators must register before participating in ranked battles.
    """
    try:
        operator = await register_operator(
            display_name=request.display_name,
            email=request.email,
        )
    except DuplicateOperatorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except OperatorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return OperatorRegisterResponse(
        operator_id=operator.operator_id,
        display_name=operator.display_name,
        email=operator.email,
        max_agents=operator.max_agents,
        max_battles_per_day=operator.max_battles_per_day,
        created_at=operator.created_at,
    )


@router.get("/operators/{operator_id}")
async def operators_get(
    operator_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get operator details by ID."""
    operator = await get_operator_by_id(operator_id)
    if not operator:
        raise HTTPException(status_code=404, detail="Operator not found")

    return {
        "operator_id": operator.operator_id,
        "display_name": operator.display_name,
        "agent_count": len(operator.agent_target_ids),
        "max_agents": operator.max_agents,
        "max_battles_per_day": operator.max_battles_per_day,
        "status": operator.status.value,
        "created_at": operator.created_at,
    }


@router.post("/operators/{operator_id}/agents")
async def operators_add_agent(
    operator_id: str,
    request: AddAgentRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """Assign an MCP server (by URL) to an operator.

    Each operator may own up to max_agents agents (default 5).
    """
    try:
        operator = await add_agent_to_operator(operator_id, request.agent_url)
    except OperatorNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MaxAgentsExceededError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except OperatorError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "operator_id": operator.operator_id,
        "agent_count": len(operator.agent_target_ids),
        "max_agents": operator.max_agents,
        "status": operator.status.value,
    }


@router.get("/operators/agent/{target_id}")
async def operators_lookup_by_agent(
    target_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Find which operator owns a given agent target_id (or null)."""
    operator = await get_operator_for_agent(target_id)
    if not operator:
        return {"operator": None, "target_id": target_id}

    return {
        "target_id": target_id,
        "operator": {
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "status": operator.status.value,
        },
    }
