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


def extract_base_template(compiled_prompt: str) -> str:
    """Extract the base system template from a fully compiled prompt.

    The base template is the first section before any ``---`` separator.
    This is the inverse of the first step in :func:`build_system_prompt`.
    """
    return compiled_prompt.split(_SECTION_SEP, 1)[0].strip()


def extract_section(compiled_prompt: str, header: str) -> str:
    """Extract the body of a named section from a compiled prompt.

    Looks for a section starting with ``[header]\\n`` among the ``---``
    separated blocks and returns its body text (everything after the
    header line).  Returns an empty string when the section is absent.
    """
    prefix = f"[{header}]\n"
    for block in compiled_prompt.split(_SECTION_SEP):
        if block.startswith(prefix):
            return block[len(prefix):].strip()
    return ""


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
