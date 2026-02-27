"""Deterministic policy guardrails: prompt injection, unsafe role change, risky capability.

All three checks are purely regex-based — no model call, no async I/O.
They are designed to run synchronously before the orchestrator's main pipeline.

Public surface:
    check_prompt_injection   — detect instruction-injection attempts
    check_unsafe_role_change — detect privilege-escalation phrasing
    check_risky_capability   — detect out-of-scope capability requests
"""

from __future__ import annotations

from typing import Final

from tdbot.policy.schemas import GuardrailResult
from tdbot.signals import Signal, compile_signals, score_signals

# ---------------------------------------------------------------------------
# Prompt injection signals
#
# These cover techniques that go beyond the behavior detector's safety signals:
# markup-based injection (XML/JSON-like system tags), role-prefix spoofing,
# and delimiter injection (triple backtick, ### headers, etc.).
# ---------------------------------------------------------------------------

_INJECTION_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Markup / delimited system tag injection
        (r"<\s*(system|instruction|prompt|context|admin)\s*>", 0.9),
        (r"\[\s*SYSTEM\s*\]", 0.9),
        (r"\[\s*INST\s*\]", 0.9),
        (r"###\s*(System|Instruction|Context|Override|Prompt)\b", 0.85),
        (r'```\s*(system|json|yaml)\s*\n.*?(override|instruction|prompt)', 0.8),
        # Role-prefix spoofing ("SYSTEM:", "ASSISTANT:", "ADMIN:")
        (r"^(SYSTEM|ADMIN|ASSISTANT|DEVELOPER|ROOT)\s*:", 0.9),
        (r"\n(SYSTEM|ADMIN|ASSISTANT|DEVELOPER|ROOT)\s*:", 0.9),
        # Direct instruction override phrasing (supplement detector signals)
        (
            r"\bignore\b.{0,30}\b(previous|above|prior|all)\b.{0,20}\b"
            r"(instructions?|prompts?)\b",
            0.85,
        ),
        (
            r"\b(игнорируй|забудь)\b.{0,30}\b(предыдущие|все)\b.{0,20}\b"
            r"(инструкции|правила|контекст|промпт)\b",
            0.85,
        ),
        (r"\bnew\b.{0,15}\binstructions?\b.{0,20}\b(are|follow|override)\b", 0.8),
        (r"\bновые\b.{0,15}\bинструкц\w*\b.{0,20}\b(следуют|переопределяют)\b", 0.8),
        (r"\boverwrite\b.{0,20}\b(system|prompt|instructions?|context)\b", 0.85),
        (r"\bперепиши\b.{0,20}\b(системный|промпт|инструкц|контекст)\w*", 0.85),
        (r"\breplace\b.{0,20}\b(system prompt|instructions?|persona)\b", 0.8),
        (r"\bзамени\b.{0,20}\b(системный промпт|инструкц|персону)\b", 0.8),
        # Prompt leakage fishing
        (r"\brepeat\b.{0,25}\b(your|the)\b.{0,20}\b(system prompt|instructions?|context)\b", 0.75),
        (r"\bprint\b.{0,20}\b(your|the)\b.{0,20}\b(system prompt|instructions?)\b", 0.75),
        (r"\bshow\b.{0,20}\b(your|the)\b.{0,20}\b(system prompt|hidden instructions?)\b", 0.75),
        (r"\bпокажи\b.{0,25}\b(системный промпт|скрытые инструкции|контекст)\b", 0.75),
        # Token smuggling keywords
        (r"\btoken\b.{0,20}\bsmuggling\b", 0.95),
        (r"\bprompt\b.{0,15}\binjection\b", 0.95),
        (r"\bprompt\b.{0,15}\bleak\b", 0.9),
        (r"\bпромпт\b.{0,15}\bинъекц\w*\b", 0.95),
        (r"\bутечк\w*\b.{0,20}\bпромпт\w*\b", 0.9),
    ]
)


def check_prompt_injection(text: str) -> GuardrailResult:
    """Return a blocked GuardrailResult if *text* contains prompt-injection patterns.

    Args:
        text: Raw user message text.

    Returns:
        :class:`~tdbot.policy.schemas.GuardrailResult` with ``allowed=False``
        when an injection signal fires, otherwise ``allowed=True``.
    """
    score = score_signals(text, _INJECTION_SIGNALS)
    if score > 0.0:
        return GuardrailResult(
            allowed=False,
            violation="prompt_injection",
            reason=(
                "Your message appears to contain an attempt to override bot instructions. "
                "If you'd like to change how I behave, try /set_tone or /set_persona."
            ),
            confidence=score,
        )
    return GuardrailResult(allowed=True, confidence=0.0)


# ---------------------------------------------------------------------------
# Unsafe role-change signals
#
# Detects attempts to assume privileged or system-level roles that would grant
# elevated permissions or bypass user-tier constraints.
# ---------------------------------------------------------------------------

_ROLE_CHANGE_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Explicit admin/developer/root role assumption
        (
            r"\b(you are|you'?re|act as|pretend (to be|you are)|become)\b.{0,30}\b"
            r"(admin|administrator|developer|root|superuser|system|owner|operator|"
            r"god mode|master|unrestricted (ai|bot|assistant))\b",
            0.9,
        ),
        (
            r"\b(ты|стань|будь|прикинься)\b.{0,30}\b"
            r"(админ(истратор)?|разработчик|root|суперпользователь|система)\b",
            0.9,
        ),
        # "Switch to developer/admin mode"
        (
            r"\b(switch|enter|enable|activate|turn on)\b.{0,25}\b"
            r"(admin|developer|root|god|debug|privileged|unrestricted)\b.{0,15}\b"
            r"(mode|role|access|capabilities?)\b",
            0.9,
        ),
        (
            r"\b(включи|активируй|переключись)\b.{0,25}\b"
            r"(режим|роль)\b.{0,15}\b(админ|разработчик|root|привилег)\w*",
            0.9,
        ),
        # "You now have admin privileges / elevated access"
        (
            r"\byou\b.{0,20}\b(now have|have been granted|are granted|have)\b.{0,25}\b"
            r"(admin|root|full|elevated|unlimited|unrestricted)\b.{0,20}\b"
            r"(access|privileges?|permissions?|rights?|capabilities?)\b",
            0.85,
        ),
        # Granting self elevated permission
        (
            r"\bgrant\b.{0,20}\b(yourself|yourself full|admin|root|elevated)\b.{0,20}\b"
            r"(access|permissions?|privileges?)\b",
            0.85,
        ),
        (
            r"\bдай\b.{0,20}\b(себе|admin|root|повышен\w*)\b.{0,20}\b"
            r"(доступ|права|привилеги)\w*",
            0.85,
        ),
        # "You are the system / the master AI" framing
        (
            r"\byou are (the )?(system|master|primary|main|controlling)\b.{0,20}\b"
            r"(ai|bot|model|agent|assistant)\b",
            0.75,
        ),
        # Explicit privilege escalation phrasing
        (r"\bescalate\b.{0,25}\bprivileges?\b", 0.9),
        (r"\bprivilege escalation\b", 0.95),
        (r"\bgain\b.{0,20}\b(root|admin|elevated)\b.{0,20}\baccess\b", 0.9),
    ]
)


def check_unsafe_role_change(text: str) -> GuardrailResult:
    """Return a blocked GuardrailResult if *text* attempts an unsafe role change.

    Args:
        text: Raw user message text.

    Returns:
        :class:`~tdbot.policy.schemas.GuardrailResult` with ``allowed=False``
        when an unsafe role change signal fires, otherwise ``allowed=True``.
    """
    score = score_signals(text, _ROLE_CHANGE_SIGNALS)
    if score > 0.0:
        return GuardrailResult(
            allowed=False,
            violation="unsafe_role_change",
            reason=(
                "I can't assume privileged or system-level roles. "
                "You can personalise my behaviour with /set_persona or /set_tone."
            ),
            confidence=score,
        )
    return GuardrailResult(allowed=True, confidence=0.0)


# ---------------------------------------------------------------------------
# Risky capability signals
#
# v1 supports prompt-based skills only (no code execution, no external access).
# These signals detect requests for capabilities that are out of scope and
# are blocked with an explanatory refusal message.
# ---------------------------------------------------------------------------

_CAPABILITY_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Code execution
        (
            r"\b(run|execute|eval|compile|invoke)\b.{0,25}\b"
            r"(code|script|program|command|shell|bash|python|javascript|sql)\b",
            0.85,
        ),
        (
            r"\b(запусти|выполни|скомпилируй)\b.{0,25}\b"
            r"(код|скрипт|программу|команду|bash|python|javascript|sql)\b",
            0.85,
        ),
        (r"\bos\.system\b|subprocess\.|exec\(|eval\(", 0.95),
        # File system access
        (
            r"\b(read|write|delete|access|open|create|modify)\b.{0,25}\b"
            r"(file|directory|folder|disk|filesystem|path)\b",
            0.75,
        ),
        (
            r"\b(прочитай|запиши|удали|открой|создай|измени)\b.{0,25}\b"
            r"(файл|директори|папк|диск|файлов\w*\s+систем\w*|путь)\w*",
            0.75,
        ),
        (r"\b(rm|mv|cp|mkdir|chmod|chown)\b.{0,20}\b/", 0.9),
        # Network / internet access
        (
            r"\b(browse|visit|fetch|scrape|crawl|download)\b.{0,25}\b"
            r"(the internet|the web|websites?|urls?|web pages?|online)\b",
            0.8,
        ),
        (
            r"\b(зайди|скачай|получи|собери|спарси|пройдись)\b.{0,25}\b"
            r"(интернет|веб|сайт|url|страниц\w*|онлайн)\b",
            0.8,
        ),
        (r"\bhttp(s?):\/\/", 0.3),
        (r"\b(send|make)\b.{0,20}\b(http|api|web)\b.{0,20}\brequest\b", 0.8),
        # Email / messaging
        (
            r"\b(send|compose|write|draft)\b.{0,20}\b"
            r"(an? email|emails?|message|messages?|sms|text)\b.{0,20}\b"
            r"(to|for)\b.{0,30}\b@\w+\b",
            0.85,
        ),
        (
            r"\b(send|forward|relay)\b.{0,20}\b(this|my|a)\b.{0,20}\b"
            r"(message|conversation|chat|email)\b.{0,20}\b(to|for)\b",
            0.7,
        ),
        # Database / storage manipulation
        (
            r"\b(drop|truncate|delete from|insert into|update)\b.{0,30}\b"
            r"(table|database|collection|schema)\b",
            0.9,
        ),
        # System / process control
        (
            r"\b(kill|stop|restart|shutdown|reboot)\b.{0,20}\b"
            r"(process|service|server|system|daemon)\b",
            0.85,
        ),
    ]
)

# Minimum score to consider a capability request risky.
_CAPABILITY_THRESHOLD: Final[float] = 0.6


def check_risky_capability(text: str) -> GuardrailResult:
    """Return a blocked GuardrailResult if *text* requests an out-of-scope capability.

    v1 of TdBot supports prompt-based skills only.  Requests for code execution,
    file access, internet browsing, email sending, or database manipulation are
    flagged as risky capabilities and blocked with an explanatory message.

    Args:
        text: Raw user message text.

    Returns:
        :class:`~tdbot.policy.schemas.GuardrailResult` with ``allowed=False``
        when a risky capability signal fires above threshold, else ``allowed=True``.
    """
    score = score_signals(text, _CAPABILITY_SIGNALS)
    if score >= _CAPABILITY_THRESHOLD:
        return GuardrailResult(
            allowed=False,
            violation="risky_capability",
            reason=(
                "That capability isn't available — I work with text and prompts only. "
                "I can't execute code, access files, browse the web, or send messages "
                "to external services."
            ),
            confidence=score,
        )
    return GuardrailResult(allowed=True, confidence=score)
