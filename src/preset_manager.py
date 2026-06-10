import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PresetManager:
    DEFAULT_PRESETS = {
        "code_analyze": {
            "name": "Code Analyze",
            "temperature": 0.2,
            "max_tokens": 8000,
            "system_prompt": """You are a code analyzer. 
ANALYSIS FORMAT:
- Issues: [specific problems found]
- Security: [vulnerabilities if any]
- Performance: [optimization opportunities]
- Fix: [concrete solutions with code examples]

Start directly with findings. No preamble. If input isn't code, state: "Input is not code. Please provide code to analyze."
"""
        },
        "brainstorm": {
            "name": "Brainstorm",
            "temperature": 0.9,
            "max_tokens": 4096,
            "system_prompt": """You are a creative ideation assistant focused on divergent thinking.

Generate diverse, unexpected ideas that span from practical to experimental. 
- Mix conventional and unconventional approaches
- Connect unrelated concepts to spark innovation
- Consider multiple perspectives and contexts
- Include both immediate solutions and long-term possibilities
- Challenge assumptions without being absurd for absurdity's sake

Structure ideas clearly but allow creative freedom in presentation. Aim for quantity and variety over filtering.
"""
        },
        "reason": {
            "name": "Reason",
            "temperature": 0.3,
            "max_tokens": 6000,
            "system_prompt": """You are a systematic reasoning assistant.

Structure all responses using clear logical progression:
1. Identify key components of the question
2. State relevant principles or facts
3. Build argument step by step
4. Address potential counterarguments
5. Conclude with justified answer

Use precise language. Show causal relationships explicitly. Quantify uncertainty where applicable.
"""
        },
    }
    
    DEFAULT_CUSTOM = {
        "name": "Custom",
        "temperature": 1.0,
        "max_tokens": 0,
        "system_prompt": "",
        "inject_prefix": "",
        "inject_suffix": "",
        "enabled": False,
    }
    
    def __init__(self, data_dir: str):
        self.presets_file = os.path.join(data_dir, "presets.json")
        self.presets = self.load()
    
    def _ensure_user_namespace(self, username: str) -> dict:
        """Ensure a user namespace exists under _users, returning it."""
        if "_users" not in self.presets:
            self.presets["_users"] = {}
        if username not in self.presets["_users"]:
            self.presets["_users"][username] = {
                "custom": dict(self.DEFAULT_CUSTOM),
                "user_templates": [],
                "group_presets": [],
            }
        return self.presets["_users"][username]
    
    def load(self) -> Dict[str, Any]:
        """Load presets from file, creating defaults if needed"""
        if not os.path.exists(self.presets_file):
            defaults = dict(self.DEFAULT_PRESETS)
            defaults["_users"] = {}
            self.save(defaults)
            return defaults.copy()
        
        try:
            with open(self.presets_file, 'r', encoding="utf-8") as f:
                presets = json.load(f)
            if not isinstance(presets, dict):
                logger.error("Error loading presets: expected an object")
                return self.DEFAULT_PRESETS.copy()

            # Migrate old flat structure to per-user namespaced (one-time)
            if "_users" not in presets:
                old_custom = presets.pop("custom", dict(self.DEFAULT_CUSTOM))
                old_templates = presets.pop("user_templates", [])
                old_groups = presets.pop("group_presets", [])
                presets["_users"] = {
                    "lumi": {
                        "custom": old_custom,
                        "user_templates": old_templates,
                        "group_presets": old_groups,
                    }
                }
                logger.info("Migrated presets to per-user namespaced format")

            # Heal legacy custom preset (disable empty default custom)
            for username, user_data in presets.get("_users", {}).items():
                custom = user_data.get("custom") if isinstance(user_data, dict) else None
                if isinstance(custom, dict) and "enabled" not in custom:
                    legacy_prompt = "You are a helpful, balanced assistant. Match your response style to the user's needs."
                    if (
                        custom.get("name") == "Custom"
                        and not custom.get("character_name")
                        and custom.get("system_prompt") == legacy_prompt
                    ):
                        custom["enabled"] = False
                        custom["system_prompt"] = ""
                        custom["temperature"] = 1.0
                        custom["max_tokens"] = 0
                        custom.setdefault("inject_prefix", "")
                        custom.setdefault("inject_suffix", "")

            # Heal missing built-in presets at root level
            if any(k not in presets for k in self.DEFAULT_PRESETS):
                presets = {**self.DEFAULT_PRESETS, **presets}

            self.save(presets)
            return presets
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
            defaults = dict(self.DEFAULT_PRESETS)
            defaults["_users"] = {}
            return defaults
    
    def save(self, presets: Dict[str, Any]) -> bool:
        """Save presets to file"""
        try:
            # Atomic write (tmp file + os.replace) so a crash or serialization
            # error mid-write can't truncate presets.json and lose every saved
            # preset. Lazy import keeps this module free of the heavy core
            # package import graph at load time.
            from core.atomic_io import atomic_write_json
            atomic_write_json(self.presets_file, presets, indent=2)
            self.presets = presets
            return True
        except Exception as e:
            logger.error(f"Error saving presets: {e}")
            return False
    
    def get(self, preset_id: str, username: Optional[str] = None) -> Dict[str, Any]:
        """Get a specific preset. For 'custom', returns user-scoped version."""
        if preset_id == "custom" and username:
            ns = self._ensure_user_namespace(username)
            return ns.get("custom", dict(self.DEFAULT_CUSTOM))
        return self.presets.get(preset_id)
    
    def get_all_for_user(self, username: str) -> Dict[str, Any]:
        """Get all presets merged with user's custom preset."""
        result = {}
        # Copy global defaults
        for k in self.DEFAULT_PRESETS:
            if k in self.presets:
                result[k] = self.presets[k]
        # Add user's custom preset
        ns = self._ensure_user_namespace(username)
        result["custom"] = ns.get("custom", dict(self.DEFAULT_CUSTOM))
        return result
    
    def get_all(self) -> Dict[str, Any]:
        """Get all presets (legacy — prefer get_all_for_user)."""
        return self.presets.copy()
    
    def update_custom(
        self,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        name: str = "",
        enabled: bool = True,
        inject_prefix: str = "",
        inject_suffix: str = "",
        character_sheet: Optional[dict] = None,
        username: Optional[str] = None,
    ) -> bool:
        """Update the custom preset for a specific user (or global fallback)."""
        preset = {
            "name": name or "Custom",
            "character_name": name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "inject_prefix": inject_prefix,
            "inject_suffix": inject_suffix,
            "enabled": enabled,
        }
        if character_sheet is not None:
            preset["character_sheet"] = character_sheet
        
        if username:
            ns = self._ensure_user_namespace(username)
            ns["custom"] = preset
        else:
            self.presets["custom"] = preset
        
        return self.save(self.presets)
    
    def get_user_templates(self, username: Optional[str] = None) -> list:
        """Get user-saved character templates, scoped to username."""
        if username:
            ns = self._ensure_user_namespace(username)
            return ns.get("user_templates", [])
        return self.presets.get("user_templates", [])
    
    def save_user_template(self, template: dict, username: Optional[str] = None) -> bool:
        """Save a new user template or update existing by id, scoped to username."""
        if username:
            ns = self._ensure_user_namespace(username)
            templates = ns.get("user_templates", [])
        else:
            templates = self.presets.get("user_templates", [])
        
        existing = next((i for i, t in enumerate(templates) if t.get("id") == template.get("id")), None)
        if existing is not None:
            templates[existing] = template
        else:
            templates.append(template)
        
        if username:
            ns["user_templates"] = templates
        else:
            self.presets["user_templates"] = templates
        
        return self.save(self.presets)
    
    def delete_user_template(self, template_id: str, username: Optional[str] = None) -> bool:
        """Delete a user template by id, scoped to username."""
        if username:
            ns = self._ensure_user_namespace(username)
            templates = ns.get("user_templates", [])
            ns["user_templates"] = [t for t in templates if t.get("id") != template_id]
        else:
            templates = self.presets.get("user_templates", [])
            self.presets["user_templates"] = [t for t in templates if t.get("id") != template_id]
        
        return self.save(self.presets)
    
    def get_group_presets(self, username: Optional[str] = None) -> list:
        """Get saved group chat presets, scoped to username."""
        if username:
            ns = self._ensure_user_namespace(username)
            return ns.get("group_presets", [])
        return self.presets.get("group_presets", [])
    
    def save_group_presets(self, groups: list, username: Optional[str] = None) -> bool:
        """Save group chat presets, scoped to username."""
        if username:
            ns = self._ensure_user_namespace(username)
            ns["group_presets"] = groups
        else:
            self.presets["group_presets"] = groups
        
        return self.save(self.presets)
