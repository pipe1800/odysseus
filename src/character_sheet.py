# src/character_sheet.py
"""
CharacterSheet model and system-prompt compiler.

A CharacterSheet is a structured 10-category JSON payload that describes
an AI persona. The compiler turns it into a compact, directive system
prompt suitable for injection as the first system message.
"""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


# ── Sub-models for each category ────────────────────────────────────────

class Identity(BaseModel):
    pronouns: str = ""
    age_range: str = ""
    species_type: str = ""
    role_occupation: str = ""
    archetype: str = ""


class Appearance(BaseModel):
    physical_description: str = ""
    typical_attire: str = ""
    first_impression: str = ""
    defining_features: List[str] = Field(default_factory=list)


class Personality(BaseModel):
    traits: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    desires_goals: str = ""
    internal_conflict: str = ""
    mbti_enneagram: str = ""
    alignment: str = ""


class Voice(BaseModel):
    tone_default: str = ""
    vocabulary_level: str = ""
    sentence_style: str = ""
    catchphrases_verbal_tics: List[str] = Field(default_factory=list)
    nonverbal_habits: str = ""
    emotional_range_in_speech: str = ""


class Background(BaseModel):
    backstory: str = ""
    current_situation: str = ""
    defining_events: List[str] = Field(default_factory=list)


class Knowledge(BaseModel):
    expertise: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    knowledge_boundaries: str = ""


class Relationships(BaseModel):
    allies: List[str] = Field(default_factory=list)
    rivals_enemies: List[str] = Field(default_factory=list)
    social_standing: str = ""
    group_role: str = ""


class Behavior(BaseModel):
    habits_routines: List[str] = Field(default_factory=list)
    stress_response: str = ""
    conflict_style: str = ""
    decision_style: str = ""
    boundaries: str = ""


class Quirks(BaseModel):
    quirks: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)


class DynamicState(BaseModel):
    current_mood: str = ""
    current_goal: str = ""
    recent_events: str = ""


# ── Top-level model ─────────────────────────────────────────────────────

class CharacterSheet(BaseModel):
    """Full character sheet matching the 10-category frontend schema."""
    identity: Identity = Field(default_factory=Identity)
    appearance: Appearance = Field(default_factory=Appearance)
    personality: Personality = Field(default_factory=Personality)
    voice: Voice = Field(default_factory=Voice)
    background: Background = Field(default_factory=Background)
    knowledge: Knowledge = Field(default_factory=Knowledge)
    relationships: Relationships = Field(default_factory=Relationships)
    behavior: Behavior = Field(default_factory=Behavior)
    quirks: Quirks = Field(default_factory=Quirks)
    dynamic_state: DynamicState = Field(default_factory=DynamicState)

    @classmethod
    def from_dict(cls, data: dict) -> CharacterSheet:
        """Construct from a raw dict, tolerating missing/extra keys."""
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return self.model_dump()


# ── Compiler ────────────────────────────────────────────────────────────

def _or_empty(val) -> str:
    """Return str of val or empty."""
    return str(val).strip() if val else ""


def _list_str(items: List[str]) -> str:
    """Comma-separated list."""
    return ", ".join(items) if items else ""


def compile_character_sheet(sheet: CharacterSheet, name: str = "") -> str:
    """Compile a CharacterSheet into a system prompt string.

    The output is a directive block that tells the model *who it is* and
    *how to behave*. Empty fields are omitted.
    """
    parts: list[str] = []

    # ── Identity header ──
    if name:
        parts.append(f"Your name is {name}.")
    elif sheet.identity.role_occupation:
        parts.append(f"You are: {sheet.identity.role_occupation}.")
    else:
        parts.append("You are a distinct character. Embody this persona fully.")

    # ── Identity ──
    ident_lines: list[str] = []
    i = sheet.identity
    if i.pronouns:
        ident_lines.append(f"Pronouns: {i.pronouns}")
    if i.age_range:
        ident_lines.append(f"Age: {i.age_range}")
    if i.species_type:
        ident_lines.append(f"Species/Type: {i.species_type}")
    if i.role_occupation:
        ident_lines.append(f"Role: {i.role_occupation}")
    if i.archetype:
        ident_lines.append(f"Archetype: {i.archetype}")
    if ident_lines:
        parts.append("Identity: " + "; ".join(ident_lines))

    # ── Appearance ──
    app_lines: list[str] = []
    a = sheet.appearance
    if a.physical_description:
        app_lines.append(a.physical_description)
    if a.typical_attire:
        app_lines.append(f"Typically wears: {a.typical_attire}")
    if a.first_impression:
        app_lines.append(f"First impression: {a.first_impression}")
    if a.defining_features:
        app_lines.append(f"Defining features: {_list_str(a.defining_features)}")
    if app_lines:
        parts.append("Appearance: " + "; ".join(app_lines))

    # ── Personality ──
    pers_lines: list[str] = []
    p = sheet.personality
    if p.traits:
        pers_lines.append(f"Traits: {_list_str(p.traits)}")
    if p.values:
        pers_lines.append(f"Values: {_list_str(p.values)}")
    if p.fears:
        pers_lines.append(f"Fears: {_list_str(p.fears)}")
    if p.desires_goals:
        pers_lines.append(f"Desires/Goals: {p.desires_goals}")
    if p.internal_conflict:
        pers_lines.append(f"Internal conflict: {p.internal_conflict}")
    if p.mbti_enneagram:
        pers_lines.append(f"MBTI/Enneagram: {p.mbti_enneagram}")
    if p.alignment:
        pers_lines.append(f"Alignment: {p.alignment}")
    if pers_lines:
        parts.append("Personality: " + "; ".join(pers_lines))

    # ── Voice ──
    voice_lines: list[str] = []
    v = sheet.voice
    if v.tone_default:
        voice_lines.append(f"Default tone: {v.tone_default}")
    if v.vocabulary_level:
        voice_lines.append(f"Vocabulary: {v.vocabulary_level}")
    if v.sentence_style:
        voice_lines.append(f"Sentence style: {v.sentence_style}")
    if v.catchphrases_verbal_tics:
        voice_lines.append(f"Catchphrases/tics: {_list_str(v.catchphrases_verbal_tics)}")
    if v.nonverbal_habits:
        voice_lines.append(f"Nonverbal habits: {v.nonverbal_habits}")
    if v.emotional_range_in_speech:
        voice_lines.append(f"Emotional range: {v.emotional_range_in_speech}")
    if voice_lines:
        parts.append("Voice & Speech: " + "; ".join(voice_lines))

    # ── Background ──
    bg_lines: list[str] = []
    b = sheet.background
    if b.backstory:
        bg_lines.append(b.backstory)
    if b.current_situation:
        bg_lines.append(f"Current situation: {b.current_situation}")
    if b.defining_events:
        bg_lines.append(f"Defining events: {_list_str(b.defining_events)}")
    if bg_lines:
        parts.append("Background: " + "; ".join(bg_lines))

    # ── Knowledge ──
    kn_lines: list[str] = []
    k = sheet.knowledge
    if k.expertise:
        kn_lines.append(f"Expertise: {_list_str(k.expertise)}")
    if k.limitations:
        kn_lines.append(f"Limitations: {_list_str(k.limitations)}")
    if k.knowledge_boundaries:
        kn_lines.append(f"Boundaries: {k.knowledge_boundaries}")
    if kn_lines:
        parts.append("Knowledge: " + "; ".join(kn_lines))

    # ── Relationships ──
    rel_lines: list[str] = []
    r = sheet.relationships
    if r.allies:
        rel_lines.append(f"Allies: {_list_str(r.allies)}")
    if r.rivals_enemies:
        rel_lines.append(f"Rivals/Enemies: {_list_str(r.rivals_enemies)}")
    if r.social_standing:
        rel_lines.append(f"Social standing: {r.social_standing}")
    if r.group_role:
        rel_lines.append(f"Group role: {r.group_role}")
    if rel_lines:
        parts.append("Relationships: " + "; ".join(rel_lines))

    # ── Behavior ──
    beh_lines: list[str] = []
    bh = sheet.behavior
    if bh.habits_routines:
        beh_lines.append(f"Habits/Routines: {_list_str(bh.habits_routines)}")
    if bh.stress_response:
        beh_lines.append(f"Under stress: {bh.stress_response}")
    if bh.conflict_style:
        beh_lines.append(f"Conflict style: {bh.conflict_style}")
    if bh.decision_style:
        beh_lines.append(f"Decision style: {bh.decision_style}")
    if bh.boundaries:
        beh_lines.append(f"Boundaries: {bh.boundaries}")
    if beh_lines:
        parts.append("Behavior: " + "; ".join(beh_lines))

    # ── Quirks ──
    qs_lines: list[str] = []
    q = sheet.quirks
    if q.quirks:
        qs_lines.append(f"Quirks: {_list_str(q.quirks)}")
    if q.likes:
        qs_lines.append(f"Likes: {_list_str(q.likes)}")
    if q.dislikes:
        qs_lines.append(f"Dislikes: {_list_str(q.dislikes)}")
    if q.secrets:
        qs_lines.append(f"Secrets: {_list_str(q.secrets)}")
    if q.hobbies:
        qs_lines.append(f"Hobbies: {_list_str(q.hobbies)}")
    if qs_lines:
        parts.append("Quirks: " + "; ".join(qs_lines))

    # ── Dynamic state ──
    dyn_lines: list[str] = []
    d = sheet.dynamic_state
    if d.current_mood:
        dyn_lines.append(f"Current mood: {d.current_mood}")
    if d.current_goal:
        dyn_lines.append(f"Current goal: {d.current_goal}")
    if d.recent_events:
        dyn_lines.append(f"Recent events: {d.recent_events}")
    if dyn_lines:
        parts.append("Current state: " + "; ".join(dyn_lines))

    # ── Core directive ──
    parts.append(
        "Stay in character at all times. Think, speak, and react as this persona. "
        "Do not break the fourth wall or acknowledge you are an AI unless the "
        "context explicitly demands it."
    )

    return "\n\n".join(parts)


def compile_character_sheet_from_dict(data: dict, name: str = "") -> str:
    """Convenience: validate + compile from a raw dict."""
    sheet = CharacterSheet.from_dict(data)
    return compile_character_sheet(sheet, name=name)
