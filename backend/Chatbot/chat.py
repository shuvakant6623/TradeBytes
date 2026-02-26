"""Chat API Router"""
import logging
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.llm_service import ollama_service
from services.memory_manager import memory_manager
from services.prompt_builder import build_full_prompt
from core.config import settings

logger = logging.getLogger("finai.chat")
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Retrieve session history from memory
    2. Build context-injected prompt
    3. Call Ollama LLM service
    4. Store new turns in memory
    5. Return structured response
    """
    logger.info(f"Chat request | session={request.session_id} | msg_len={len(request.message)}")

    # Guard: basic content filter
    if len(request.message.strip()) < 2:
        raise HTTPException(status_code=400, detail="Message too short")

    # 1. Get session history
    history_text = memory_manager.get_history_as_text(request.session_id)

    # 2. Build prompt
    system_prompt, user_prompt = build_full_prompt(
        user_message=request.message,
        conversation_history=history_text,
        portfolio=request.portfolio,
        user_profile=request.user_profile,
        market_context=request.market_context,
    )

    # 3. LLM inference
    structured_response, latency_ms = await ollama_service.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # 4. Store turns in memory
    memory_manager.add_turn(request.session_id, "user", request.message)
    memory_manager.add_turn(request.session_id, "assistant", structured_response.summary)

    logger.info(f"Chat complete | session={request.session_id} | latency={latency_ms:.0f}ms | confidence={structured_response.confidence_level}")

    return ChatResponse(
        session_id=request.session_id,
        response=structured_response,
        latency_ms=latency_ms,
    )

@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    memory_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}