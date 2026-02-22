"""Prompt merge builder.

Assembles the final system prompt from its constituent segments in a
deterministic, human-readable order:

  1. Base system template (core rules, shared across all users)
  2. Persona segment (per-user tone / style overrides)
  3. Skill packs (ordered alphabetically by skill name for reproducibility)
  4. Long-term profile (compacted memory from past conversations)
  5. Short-term window (summary of the most recent conversation turns)

Callers should store the returned string as ``SnapshotRecord.system_prompt``
and pass it as ``UserContext.system_prompt`` to the inference adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tdbot.prompt.schemas import PromptComponents

# Visible separator between prompt sections — easy to scan in debug logs.
_SECTION_SEP = "\n\n---\n\n"


def build_system_prompt(components: PromptComponents) -> str:
    """Merge all prompt *components* into a single system-prompt string.

    Empty segments (blank or whitespace-only) are omitted so the assembled
    prompt does not contain spurious section headers.

    Args:
        components: The individual segments to merge.

    Returns:
        Fully assembled system-prompt text ready for the model.
    """
    sections: list[str] = [components.base_system_template.strip()]

    if components.persona_segment.strip():
        sections.append(f"[Persona]\n{components.persona_segment.strip()}")

    # Sort skill packs alphabetically so output is deterministic regardless of
    # insertion order into the dict.
    for skill_name in sorted(components.skill_packs):
        skill_prompt = components.skill_packs[skill_name].strip()
        if skill_prompt:
            sections.append(f"[Skill: {skill_name}]\n{skill_prompt}")

    if components.long_term_profile.strip():
        sections.append(f"[Long-term Profile]\n{components.long_term_profile.strip()}")

    if components.short_term_window.strip():
        sections.append(f"[Recent Context]\n{components.short_term_window.strip()}")

    return _SECTION_SEP.join(sections)
