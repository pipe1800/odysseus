"""
MemU memory bridge for Odysseus.
Talks to the local MemU server (localhost:8000) for persistent,
semantic, cross-session memory. Same backend that powers Hermes.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MEMU_BASE = "http://host.docker.internal:8000"
MEMU_AGENT_ID = "Lumi"
REQUEST_TIMEOUT = 30.0


async def retrieve_memories(query: str, user_id: str) -> str:
    """Fetch relevant memory context from MemU for a user query."""
    try:
        logger.warning(f"[memu] retrieve called: user_id={user_id!r} query={query[:60]!r}")
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{MEMU_BASE}/retrieve",
                json={
                    "user_id": user_id,
                    "query": query,
                    "agent_id": MEMU_AGENT_ID,
                    "route_intention": False,
                },
            )
            logger.warning(f"[memu] retrieve response: HTTP {resp.status_code}")
            if resp.is_success:
                data = resp.json()
                result = data.get("result", data)
                context = data.get("context", "")
                if not context:
                    parts = []
                    # Only items — categories are broad summaries, too noisy
                    for item in result.get("items", []):
                        summary = item.get("summary", "")
                        if summary:
                            parts.append(f"- {summary}")
                    context = "\n".join(parts)
                else:
                    logger.warning(f"[memu] retrieve: used server 'context' field ({len(context)} chars)")
                # Keep it surgical — small, scannable, in the system prompt
                max_ctx = 800
                if len(context) > max_ctx:
                    orig_len = len(context)
                    context = context[:max_ctx] + "\n..."
                    logger.warning(f"[memu] context trimmed from {orig_len} to {max_ctx}")
                if context and context.strip():
                    logger.warning(f"[memu] retrieve success: {len(context)} chars, items={len(result.get('items',[]))} cats={len(result.get('categories',[]))}")
                    # Log first 200 chars to see format
                    logger.warning(f"[memu] context preview: {context[:200]}")
                    return context.strip()
                logger.warning(f"[memu] retrieve: empty context")
    except Exception as e:
        logger.warning(f"[memu] retrieve FAILED: {type(e).__name__}: {e}")
    return ""


async def memorize(user_id: str, conversation: list[dict]) -> bool:
    """Store a conversation turn in MemU for future retrieval."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{MEMU_BASE}/memorize/direct",
                json={
                    "user_id": user_id,
                    "agent_id": MEMU_AGENT_ID,
                    "conversation": conversation,
                },
            )
            return resp.is_success
    except Exception as e:
        logger.warning(f"MemU memorize failed: {e}")
        return False


async def list_categories(user_id: str) -> list[dict]:
    """List all memory categories."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                f"{MEMU_BASE}/categories",
                params={"user_id": user_id, "agent_id": MEMU_AGENT_ID},
            )
            if resp.is_success:
                return resp.json()
    except Exception as e:
        logger.warning(f"MemU categories failed: {e}")
    return []


async def clear_memories(user_id: str) -> bool:
    """Clear all memories for a user."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{MEMU_BASE}/clear",
                json={"user_id": user_id, "agent_id": MEMU_AGENT_ID},
            )
            return resp.is_success
    except Exception as e:
        logger.warning(f"MemU clear failed: {e}")
        return False


def format_memory_results(categories: list[dict]) -> str:
    """Format category list for agent tool output."""
    if not categories:
        return "No memories found."
    lines = [f"Found {len(categories)} memory categories:\n"]
    for cat in categories[:20]:
        name = cat.get("name", "?")
        summary = (cat.get("summary") or cat.get("description") or "")[:120]
        item_count = cat.get("item_count", "?")
        lines.append(f"- **{name}** ({item_count} items): {summary}")
    return "\n".join(lines)
