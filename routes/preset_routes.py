"""Preset routes — /api/presets GET, /api/presets/custom POST, user templates CRUD."""

import logging
import json
import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from src.request_models import PresetUpdateRequest
from src.auth_helpers import get_current_user
from core.middleware import require_admin
from src.auth_helpers import effective_user

logger = logging.getLogger(__name__)


class UserTemplateRequest(BaseModel):
    id: str = ""
    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str = Field("", max_length=10000)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(0, ge=0, le=65536)


def setup_preset_routes(preset_manager) -> APIRouter:
    router = APIRouter(tags=["presets"])

    @router.get("/api/presets")
    async def get_presets(request: Request) -> Dict[str, Any]:
        """Get presets for the current user — global + user's custom."""
        user = get_current_user(request)
        if user:
            return preset_manager.get_all_for_user(user)
        return preset_manager.presets

    @router.post("/api/presets/custom")
    async def update_custom_preset(
        request: Request,
        preset_update: PresetUpdateRequest,
    ) -> Dict[str, Any]:
        """Update the custom preset for the current user."""
        try:
            user = get_current_user(request)
            success = preset_manager.update_custom(
                preset_update.temperature,
                preset_update.max_tokens,
                preset_update.system_prompt,
                preset_update.name,
                preset_update.enabled,
                preset_update.inject_prefix,
                preset_update.inject_suffix,
                preset_update.character_sheet,
                username=user,
            )
            if success:
                return {"success": True, "message": "Custom preset updated"}
            return {"success": False, "message": "Failed to save preset"}
        except Exception as e:
            logger.error(f"Preset update error: {e}")
            raise HTTPException(500, "Failed to update custom preset")

    @router.get("/api/presets/templates")
    async def get_user_templates(request: Request) -> List[Dict]:
        user = get_current_user(request)
        return preset_manager.get_user_templates(username=user)

    @router.post("/api/presets/templates")
    async def save_user_template(
        request: Request,
        req: UserTemplateRequest,
        _admin: None = Depends(require_admin),
    ) -> Dict[str, Any]:
        user = get_current_user(request)
        template = req.model_dump()
        if not template["id"]:
            template["id"] = f"user-{uuid.uuid4().hex[:8]}"
        success = preset_manager.save_user_template(template, username=user)
        if success:
            return {"success": True, "template": template}
        return {"success": False, "message": "Failed to save template"}

    @router.delete("/api/presets/templates/{template_id}")
    async def delete_user_template(
        request: Request,
        template_id: str,
        _admin: None = Depends(require_admin),
    ) -> Dict[str, Any]:
        user = get_current_user(request)
        success = preset_manager.delete_user_template(template_id, username=user)
        if success:
            return {"success": True}
        return {"success": False, "message": "Failed to delete template"}

    @router.post("/api/presets/expand")
    async def expand_character_prompt(request: Request) -> Dict[str, Any]:
        """Use AI to expand a rough character description into a full system prompt."""
        from src.ai_interaction import _resolve_model
        from src.llm_core import llm_call_async

        data = await request.json()
        draft = (data.get("prompt") or "").strip()
        name = (data.get("name") or "").strip()

        if not draft and not name:
            return {"success": False, "message": "Nothing to expand"}

        user_input = ""
        if name:
            user_input += f"Character name: {name}\n"
        if draft:
            user_input += f"Notes: {draft}\n"

        messages = [
            {"role": "system", "content": (
                "You are an expert at writing character system prompts for AI assistants. "
                "The user will give you a character name and/or rough notes. "
                "Write a concise, effective system prompt (3-6 sentences) that captures the character's personality, "
                "speaking style, knowledge areas, and behavioral guidelines. "
                "Output ONLY the system prompt text — no quotes, no preamble, no explanation."
            )},
            {"role": "user", "content": user_input},
        ]

        try:
            model_spec = data.get("model") or ""
            user = effective_user(request)
            url, model, headers = _resolve_model(model_spec, owner=user)
            result = await llm_call_async(url, model, messages, temperature=0.8, max_tokens=500, headers=headers)
            return {"success": True, "prompt": result.strip()}
        except Exception as e:
            logger.error(f"Expand prompt failed: {e}")
            return {"success": False, "message": str(e)}

    @router.post("/api/presets/expand-sheet")
    async def expand_character_sheet(request: Request) -> Dict[str, Any]:
        """Use AI to fill a CharacterSheet from rough notes."""
        from src.ai_interaction import _resolve_model
        from src.llm_core import llm_call_async
        from src.character_sheet import CharacterSheet

        data = await request.json()
        draft = (data.get("prompt") or "").strip()
        name = (data.get("name") or "").strip()

        if not draft and not name:
            return {"success": False, "message": "Nothing to expand"}

        # Build the schema description for the LLM
        schema_desc = """{
  "identity": {"pronouns": "", "age_range": "", "species_type": "", "role_occupation": "", "archetype": ""},
  "appearance": {"physical_description": "", "typical_attire": "", "first_impression": "", "defining_features": []},
  "personality": {"traits": [], "values": [], "fears": [], "desires_goals": "", "internal_conflict": "", "mbti_enneagram": "", "alignment": ""},
  "voice": {"tone_default": "", "vocabulary_level": "", "sentence_style": "", "catchphrases_verbal_tics": [], "nonverbal_habits": "", "emotional_range_in_speech": ""},
  "background": {"backstory": "", "current_situation": "", "defining_events": []},
  "knowledge": {"expertise": [], "limitations": [], "knowledge_boundaries": ""},
  "relationships": {"allies": [], "rivals_enemies": [], "social_standing": "", "group_role": ""},
  "behavior": {"habits_routines": [], "stress_response": "", "conflict_style": "", "decision_style": "", "boundaries": ""},
  "quirks": {"quirks": [], "likes": [], "dislikes": [], "secrets": [], "hobbies": []},
  "dynamic_state": {"current_mood": "", "current_goal": "", "recent_events": ""}
}"""

        messages = [
            {"role": "system", "content": (
                "You are an expert character designer. Given a character name and rough notes, "
                "fill out a complete character sheet in JSON format. Be creative but coherent. "
                "Fill only fields that make sense — leave irrelevant ones empty. "
                "Lists should have 2-5 items where appropriate. "
                "Output ONLY valid JSON matching this schema, no markdown, no explanation:\n\n"
                + schema_desc
            )},
            {"role": "user", "content": (
                f"Character name: {name or 'Unnamed'}\n"
                f"Notes: {draft or 'Create a compelling original character.'}\n\n"
                "Output the full character sheet as JSON."
            )},
        ]

        try:
            model_spec = data.get("model") or ""
            url, model, headers = _resolve_model(model_spec)
            result = await llm_call_async(
                url, model, messages, temperature=0.8, max_tokens=2000, headers=headers
            )
            # Parse the JSON response
            sheet_dict = json.loads(result.strip())
            # Validate through pydantic
            sheet = CharacterSheet.from_dict(sheet_dict)
            return {"success": True, "character_sheet": sheet.to_dict()}
        except json.JSONDecodeError as e:
            logger.error(f"Sheet expand JSON parse error: {e}")
            return {"success": False, "message": f"AI returned invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"Sheet expand failed: {e}")
            return {"success": False, "message": str(e)}

    # ── Group presets ──
    @router.get("/api/presets/groups")
    async def get_group_presets(request: Request):
        """Get saved group chat presets for current user."""
        user = get_current_user(request)
        return {"groups": preset_manager.get_group_presets(username=user)}

    @router.post("/api/presets/groups")
    async def save_group_presets(request: Request, _admin: None = Depends(require_admin)):
        """Save group chat presets for current user."""
        user = get_current_user(request)
        data = await request.json()
        preset_manager.save_group_presets(data.get("groups", []), username=user)
        return {"ok": True}

    return router
