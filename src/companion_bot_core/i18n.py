"""Lightweight i18n helpers for user-facing bot messages.

Supported locales:
- ``ru`` (default)
- ``en``
"""

from __future__ import annotations

from typing import Literal

Locale = Literal["ru", "en"]
SUPPORTED_LOCALES: frozenset[Locale] = frozenset({"ru", "en"})


def normalize_locale(locale: str | None) -> Locale:
    """Return a supported locale code, defaulting to ``ru``."""
    if not locale:
        return "ru"
    value = locale.strip().lower()
    if value.startswith("en"):
        return "en"
    if value.startswith("ru"):
        return "ru"
    return "ru"


_MESSAGES: dict[str, dict[Locale, str]] = {
    "start.text": {
        "ru": (
            "Привет! Я твой персональный бот-компаньон.\n\n"
            "Доступные команды:\n"
            "/profile — показать текущие настройки\n"
            "/set_language <ru|en> — сменить язык общения\n"
            "/set_tone <tone> — настроить тон (friendly, professional, playful…)\n"
            "/set_persona <name> — задать имя персоны\n"
            "/memory_compact_now — сжать историю диалога\n"
            "/reset_persona — вернуть персону по умолчанию\n"
            "/rollback — откатить профиль к предыдущей версии\n"
            "/privacy — сводка по приватности\n"
            "/delete_my_data — удалить все твои данные навсегда"
        ),
        "en": (
            "Hello! I am your personal companion bot.\n\n"
            "Available commands:\n"
            "/profile — view your current settings\n"
            "/set_language <ru|en> — change chat language\n"
            "/set_tone <tone> — adjust my tone (friendly, professional, playful…)\n"
            "/set_persona <name> — give me a persona name\n"
            "/memory_compact_now — compress your conversation history\n"
            "/reset_persona — restore default persona\n"
            "/rollback — revert to the previous prompt version\n"
            "/privacy — privacy policy summary\n"
            "/delete_my_data — permanently delete all your data"
        ),
    },
    "profile.not_set": {"ru": "(не задано)", "en": "(not set)"},
    "profile.decrypt_failed": {"ru": "(не удалось расшифровать)", "en": "(unable to decrypt)"},
    "profile.summary": {
        "ru": (
            "Telegram ID: {telegram_id}\n"
            "Статус: {status}\n"
            "Язык: {user_locale}\n"
            "Часовой пояс: {timezone}\n\n"
            "Персона: {persona}\n"
            "Тон: {tone}\n\n"
            "Используй /set_tone и /set_persona для настройки."
        ),
        "en": (
            "Telegram ID: {telegram_id}\n"
            "Status: {status}\n"
            "Locale: {user_locale}\n"
            "Timezone: {timezone}\n\n"
            "Persona: {persona}\n"
            "Tone: {tone}\n\n"
            "Use /set_tone and /set_persona to customise."
        ),
    },
    "set_language.help": {
        "ru": "Укажи язык.\nПример: /set_language ru\nПоддерживаемые: ru, en",
        "en": "Please provide a language.\nExample: /set_language en\nSupported: ru, en",
    },
    "set_language.invalid": {
        "ru": "Неизвестный язык '{value}'. Поддерживаемые: ru, en",
        "en": "Unknown language '{value}'. Supported: ru, en",
    },
    "set_language.updated": {
        "ru": "Язык общения переключён на русский.",
        "en": "Chat language has been switched to English.",
    },
    "profile.lock_in_progress": {
        "ru": "Обновление профиля уже выполняется. Попробуй ещё раз через минуту.",
        "en": "A profile update is already in progress. Please try again in a moment.",
    },
    "set_tone.missing": {
        "ru": "Укажи тон.\nПример: /set_tone friendly\nДоступные тоны: {tones}",
        "en": "Please provide a tone.\nExample: /set_tone friendly\nValid tones: {tones}",
    },
    "set_tone.invalid": {
        "ru": "Неизвестный тон '{tone}'.\nДоступные тоны: {tones}",
        "en": "Unknown tone '{tone}'.\nValid tones: {tones}",
    },
    "set_tone.updated": {
        "ru": "Тон изменён на '{tone}'.\nИзменения применятся со следующего сообщения.",
        "en": "Tone set to '{tone}'.\nChanges will be applied starting from your next message.",
    },
    "set_persona.missing": {
        "ru": "Укажи имя персоны.\nПример: /set_persona Alex",
        "en": "Please provide a persona name.\nExample: /set_persona Alex",
    },
    "set_persona.too_long": {
        "ru": "Имя персоны должно быть не длиннее 64 символов.",
        "en": "Persona name must be 64 characters or fewer.",
    },
    "set_persona.control_chars": {
        "ru": "Имя персоны не должно содержать управляющие символы.",
        "en": "Persona name must not contain control characters.",
    },
    "set_persona.updated": {
        "ru": "Имя персоны изменено на '{name}'.",
        "en": "Persona name set to '{name}'.",
    },
    "memory_compact.in_progress": {
        "ru": "Компактизация уже выполняется. Пожалуйста, подожди.",
        "en": "A compaction is already in progress. Please wait.",
    },
    "memory_compact.enqueue_failed": {
        "ru": "Не удалось поставить компактизацию в очередь. Попробуй ещё раз.",
        "en": "Failed to enqueue compaction. Please try again.",
    },
    "memory_compact.requested": {
        "ru": (
            "Компактизация памяти запрошена.\n"
            "Профиль будет обновлён на основе недавних диалогов."
        ),
        "en": (
            "Memory compaction requested.\n"
            "Your prompt profile will be refined based on recent conversations shortly."
        ),
    },
    "reset_persona.updated": {
        "ru": (
            "Персона сброшена к значениям по умолчанию.\n"
            "Используй /set_persona и /set_tone для новой настройки."
        ),
        "en": (
            "Your persona has been reset to defaults.\n"
            "Use /set_persona and /set_tone to customise again."
        ),
    },
    "rollback.updated": {
        "ru": "Профиль откатан к версии {version}.\nИзменения применятся со следующего сообщения.",
        "en": (
            "Prompt rolled back to version {version}.\n"
            "Changes will be applied starting from your next message."
        ),
    },
    "privacy.summary": {
        "ru": (
            "Сводка по приватности:\n\n"
            "• Сообщения хранятся до 7 дней для сохранения контекста.\n"
            "• Настройки профиля хранятся до запроса на удаление.\n"
            "• Данные не продаются и не передаются третьим лицам.\n\n"
            "Используй /delete_my_data для полного удаления персональных данных."
        ),
        "en": (
            "Privacy summary:\n\n"
            "• Messages are retained for up to 7 days to maintain conversation context.\n"
            "• Profile settings are stored until you request deletion.\n"
            "• No data is sold or shared with third parties.\n\n"
            "Use /delete_my_data to permanently remove all your personal data."
        ),
    },
    "delete_my_data.done": {
        "ru": (
            "Твои персональные данные удалены навсегда.\n\n"
            "История диалога, настройки профиля и данные персоны удалены. "
            "Это действие нельзя отменить."
        ),
        "en": (
            "Your personal data has been permanently deleted.\n\n"
            "Conversation history, profile settings, and persona data have been removed. "
            "This action cannot be undone."
        ),
    },
    "handle.error": {
        "ru": "Не удалось обработать сообщение. Попробуй ещё раз чуть позже.",
        "en": "Something went wrong while processing your message. Please try again in a moment.",
    },
    "notice.profile_updated": {
        "ru": "Твой профиль общения обновлён на основе недавних диалогов.",
        "en": "Your conversation profile has been updated based on recent interactions.",
    },
    "orchestrator.circuit_open": {
        "ru": "Сейчас есть проблемы с доступом к AI-сервису. Попробуй ещё раз чуть позже.",
        "en": "I'm having trouble reaching the AI service right now. Please try again in a moment.",
    },
    "orchestrator.refuse": {
        "ru": (
            "Я не могу применить это изменение — оно конфликтует с правилами безопасности. "
            "Для настройки используй /set_tone или /set_persona."
        ),
        "en": (
            "I can't make that change — it conflicts with safety guidelines. "
            "If you'd like to adjust your experience, try /set_tone or /set_persona."
        ),
    },
    "orchestrator.guardrail.prompt_injection": {
        "ru": (
            "Похоже, в сообщении есть попытка переопределить инструкции бота. "
            "Если хочешь изменить поведение, используй /set_tone или /set_persona."
        ),
        "en": (
            "Your message appears to contain an attempt to override bot instructions. "
            "If you'd like to change how I behave, try /set_tone or /set_persona."
        ),
    },
    "orchestrator.guardrail.unsafe_role_change": {
        "ru": (
            "Я не могу принимать привилегированные или системные роли. "
            "Для персонализации используй /set_persona или /set_tone."
        ),
        "en": (
            "I can't assume privileged or system-level roles. "
            "You can personalise my behaviour with /set_persona or /set_tone."
        ),
    },
    "orchestrator.guardrail.risky_capability": {
        "ru": (
            "Эта возможность недоступна — я работаю только с текстом и промптами. "
            "Я не могу запускать код, читать файлы, просматривать веб или отправлять сообщения "
            "во внешние сервисы."
        ),
        "en": (
            "That capability isn't available — I work with text and prompts only. "
            "I can't execute code, access files, browse the web, or send messages "
            "to external services."
        ),
    },
    "orchestrator.safety_fallback": {
        "ru": "Мне не удалось безопасно сформировать ответ. Попробуй переформулировать запрос.",
        "en": (
            "I wasn't able to generate a suitable response for that message. "
            "Could you try rephrasing?"
        ),
    },
    "orchestrator.change_applied": {
        "ru": "Готово! Я запомнил предпочтение и подстроюсь под него.",
        "en": "Done! I've recorded your preference and will adapt accordingly.",
    },
    "orchestrator.change_cancelled": {
        "ru": "Хорошо, оставляю всё как есть.",
        "en": "No problem, keeping things as they are.",
    },
    "orchestrator.change_apply_failed": {
        "ru": "Подтверждение получено, но не удалось извлечь настройку из исходного сообщения.",
        "en": (
            "I understood your confirmation but couldn't extract "
            "the setting from your original message."
        ),
    },
    "orchestrator.intent.persona_change": {"ru": "твою персону", "en": "your persona"},
    "orchestrator.intent.tone_change": {"ru": "твой тон", "en": "your tone"},
    "orchestrator.intent.skill_add_prompt": {"ru": "твои навыки", "en": "your skills"},
    "orchestrator.intent.skill_remove": {"ru": "твои навыки", "en": "your skills"},
    "orchestrator.confirm_template": {
        "ru": (
            "Ты хочешь изменить {label}. Это изменение средней критичности. "
            "Ответь 'да' для подтверждения или 'нет' для отмены."
        ),
        "en": (
            "You'd like to change {label}. This is a moderate setting change. "
            "Reply 'yes' to confirm or 'no' to cancel."
        ),
    },
    "orchestrator.clarification": {
        "ru": (
            "Я не до конца понял, что именно нужно изменить. "
            "Можно использовать /set_tone или /set_persona."
        ),
        "en": (
            "I'm not sure what you'd like to change. "
            "You can use /set_tone or /set_persona."
        ),
    },
    "orchestrator.abuse_block": {
        "ru": "Слишком много нарушений за короткое время. Подожди пару минут и попробуй снова.",
        "en": (
            "You've triggered too many policy violations in a short time. "
            "Please wait a few minutes before sending more messages."
        ),
    },
    "prompt.language_instruction": {
        "ru": "Отвечай на русском языке, если пользователь явно не попросит другой язык.",
        "en": "Reply in English unless the user explicitly asks for another language.",
    },
    "onboarding.question": {
        "ru": (
            "Расскажи немного о себе: чем занимаешься, что тебя интересует "
            "и как я могу тебе помочь? Я запомню это, чтобы лучше подстраиваться под тебя."
        ),
        "en": (
            "Tell me a bit about yourself: what do you do, what are your interests, "
            "and how can I help you? I'll remember this to tailor my responses to you."
        ),
    },
    "onboarding.done": {
        "ru": (
            "Отлично, запомнил! Буду учитывать это в наших разговорах.\n\n"
            "Можешь написать мне что угодно — я здесь, чтобы помочь."
        ),
        "en": (
            "Got it, thanks! I'll keep that in mind for our conversations.\n\n"
            "Feel free to send me anything — I'm here to help."
        ),
    },
}


def tr(key: str, locale: str | None = None, **kwargs: object) -> str:
    """Translate *key* for *locale* and format with ``kwargs``."""
    resolved = normalize_locale(locale)
    values = _MESSAGES[key]
    template = values.get(resolved, values["ru"])
    return template.format(**kwargs)
