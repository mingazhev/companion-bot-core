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
    "profile.not_set": {"ru": "(не задано)", "en": "(not set)"},
    "profile.decrypt_failed": {"ru": "(не удалось расшифровать)", "en": "(unable to decrypt)"},
    "profile.summary": {
        "ru": (
            "Telegram ID: {telegram_id}\n"
            "Статус: {status}\n"
            "Язык: {user_locale}\n\n"
            "Персона: {persona}\n"
            "Тон: {tone}\n\n"
            "Используй /set_tone и /set_persona для настройки."
        ),
        "en": (
            "Telegram ID: {telegram_id}\n"
            "Status: {status}\n"
            "Locale: {user_locale}\n\n"
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
        "ru": "Имя персоны изменено на '{name}'.\nИзменения применятся со следующего сообщения.",
        "en": "Persona name set to '{name}'.\nChanges will take effect from your next message.",
    },
    "reset_persona.updated": {
        "ru": (
            "Персона и тон общения сброшены к настройкам по умолчанию.\n"
            "Используй /set_persona и /set_tone для новой настройки."
        ),
        "en": (
            "Persona and tone have been reset to defaults.\n"
            "Use /set_persona and /set_tone to customise again."
        ),
    },
    "rollback.no_snapshot": {
        "ru": "Нечего откатывать — у тебя пока нет сохранённого профиля.",
        "en": "Nothing to roll back — you don't have a saved profile yet.",
    },
    "rollback.no_previous": {
        "ru": "Откатываться некуда — это единственная версия профиля.",
        "en": "Nothing to roll back to — this is the only profile version.",
    },
    "rollback.updated": {
        "ru": "Профиль откатан к версии {version}.\nИзменения применятся со следующего сообщения.",
        "en": (
            "Profile rolled back to version {version}.\n"
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
    "delete_my_data.confirm_prompt": {
        "ru": (
            "Ты уверен? Это действие удалит все твои данные навсегда: "
            "историю диалогов, профиль, персону и навыки.\n\n"
            "Отменить это будет невозможно."
        ),
        "en": (
            "Are you sure? This will permanently delete all your data: "
            "conversation history, profile, persona, and skills.\n\n"
            "This cannot be undone."
        ),
    },
    "delete_my_data.done": {
        "ru": "Все твои данные удалены навсегда.",
        "en": "All your data has been permanently deleted.",
    },
    "delete_my_data.cancelled": {
        "ru": "Удаление отменено. Твои данные в безопасности.",
        "en": "Deletion cancelled. Your data is safe.",
    },
    "delete_my_data.expired": {
        "ru": "Время подтверждения истекло. Используй /delete_my_data снова.",
        "en": "Confirmation expired. Use /delete_my_data again.",
    },
    "handle.error": {
        "ru": "Не удалось обработать сообщение. Попробуй ещё раз чуть позже.",
        "en": "Something went wrong while processing your message. Please try again in a moment.",
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
    "orchestrator.pending_cancelled": {
        "ru": "Предыдущий запрос на изменение отменён.",
        "en": "Previous change request cancelled.",
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
    "orchestrator.auto_applied_tone": {
        "ru": "Кстати, я обновил тон общения.",
        "en": "By the way, I've updated the conversation tone.",
    },
    "orchestrator.auto_applied_persona": {
        "ru": "Кстати, я обновил персону.",
        "en": "By the way, I've updated the persona.",
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
    # --- /memory, /remember, /forget ---
    "memory.header": {
        "ru": "Вот что я о тебе помню:\n",
        "en": "Here's what I remember about you:\n",
    },
    "memory.persona": {
        "ru": "Персона: {value}",
        "en": "Persona: {value}",
    },
    "memory.tone": {
        "ru": "Тон: {value}",
        "en": "Tone: {value}",
    },
    "memory.skills": {
        "ru": "Навыки: {value}",
        "en": "Skills: {value}",
    },
    "memory.profile": {
        "ru": "Что я запомнил о тебе:\n{value}",
        "en": "What I remember about you:\n{value}",
    },
    "memory.empty": {
        "ru": (
            "Я пока ничего о тебе не знаю. "
            "Просто общайся со мной, и я постепенно узнаю тебя лучше.\n\n"
            "Или используй /remember, чтобы рассказать мне что-то о себе."
        ),
        "en": (
            "I don't know anything about you yet. "
            "Just chat with me and I'll gradually get to know you.\n\n"
            "Or use /remember to tell me something about yourself."
        ),
    },
    "remember.missing": {
        "ru": (
            "Расскажи, что запомнить.\n"
            "Пример: /remember люблю кофе по утрам"
        ),
        "en": (
            "Tell me what to remember.\n"
            "Example: /remember I love coffee in the morning"
        ),
    },
    "remember.saved": {
        "ru": "Запомнил: {fact}",
        "en": "Got it, I'll remember: {fact}",
    },
    "remember.truncated": {
        "ru": "Текст был слишком длинным — я сохраню первые 500 символов.",
        "en": "That was a bit long — I'll save the first 500 characters.",
    },
    "forget.missing": {
        "ru": (
            "Укажи, что забыть.\n"
            "Пример: /forget кофе"
        ),
        "en": (
            "Tell me what to forget.\n"
            "Example: /forget coffee"
        ),
    },
    "forget.not_found": {
        "ru": "Не нашёл ничего похожего в своей памяти.",
        "en": "I couldn't find anything matching that in my memory.",
    },
    "forget.done": {
        "ru": "Забыл: {fact}",
        "en": "Forgotten: {fact}",
    },
    # --- /help ---
    "help.text": {
        "ru": (
            "Доступные команды:\n\n"
            "/help — эта справка\n"
            "/privacy — сводка по приватности\n"
            "/set_language <ru|en> — сменить язык общения\n\n"
            "Только в личных сообщениях:\n"
            "/memory — что я о тебе помню\n"
            "/remember <факт> — рассказать мне что-то о себе\n"
            "/forget <факт> — попросить забыть что-то\n"
            "/settings — настройки (тон, персона, навыки, язык)\n"
            "/personas — выбрать персону\n"
            "/skills — каталог навыков\n"
            "/profile — твои текущие настройки\n"
            "/set_tone <тон> — настроить тон (friendly, professional, playful…)\n"
            "/set_persona <имя> — задать имя персоны\n"
            "/refresh_memory — обновить то, что я помню о тебе\n"
            "/reset_persona — вернуть персону по умолчанию\n"
            "/rollback — откатить профиль к предыдущей версии\n"
            "/delete_my_data — удалить все твои данные навсегда"
        ),
        "en": (
            "Available commands:\n\n"
            "/help — this help\n"
            "/privacy — privacy policy summary\n"
            "/set_language <ru|en> — change chat language\n\n"
            "Private chats only:\n"
            "/memory — what I remember about you\n"
            "/remember <fact> — tell me something about yourself\n"
            "/forget <fact> — ask me to forget something\n"
            "/settings — settings (tone, persona, skills, language)\n"
            "/personas — choose a persona\n"
            "/skills — skill catalog\n"
            "/profile — your current settings\n"
            "/set_tone <tone> — adjust my tone (friendly, professional, playful…)\n"
            "/set_persona <name> — give me a persona name\n"
            "/refresh_memory — refresh what I remember about you\n"
            "/reset_persona — restore default persona\n"
            "/rollback — revert to the previous profile version\n"
            "/delete_my_data — permanently delete all your data"
        ),
    },
    # --- Reworked /start ---
    "start.welcome_new": {
        "ru": (
            "Привет! Я твой персональный AI-компаньон.\n\n"
            "Я учусь понимать тебя с каждым разговором — запоминаю, что тебе важно, "
            "и подстраиваюсь под твой стиль общения.\n\n"
            "Просто напиши мне что-нибудь, и мы начнём!"
        ),
        "en": (
            "Hi there! I'm your personal AI companion.\n\n"
            "I learn about you with every conversation — I remember what matters to you "
            "and adapt to your communication style.\n\n"
            "Just send me a message and let's get started!"
        ),
    },
    "start.welcome_back": {
        "ru": "С возвращением, {name}! Чем могу помочь?",
        "en": "Welcome back, {name}! How can I help?",
    },
    "start.welcome_back_no_name": {
        "ru": "С возвращением! Чем могу помочь?",
        "en": "Welcome back! How can I help?",
    },
    # --- Reworked notice ---
    "notice.profile_updated_v2": {
        "ru": (
            "Я обновил то, что помню о тебе, на основе наших разговоров. "
            "Посмотри через /memory."
        ),
        "en": (
            "I've updated what I remember about you based on our conversations. "
            "Check it out with /memory."
        ),
    },
    "notice.facts_added": {
        "ru": "Новое:\n{items}",
        "en": "New:\n{items}",
    },
    "notice.facts_removed": {
        "ru": "Забыто:\n{items}",
        "en": "Removed:\n{items}",
    },
    "notice.persona_adjusted": {
        "ru": "Тон общения немного подстроен.",
        "en": "Communication tone slightly adjusted.",
    },
    "notice.skills_added": {
        "ru": "Добавлены навыки: {items}",
        "en": "Skills added: {items}",
    },
    "notice.skills_removed": {
        "ru": "Удалены навыки: {items}",
        "en": "Skills removed: {items}",
    },
    # --- /refresh_memory (renamed from /memory_compact_now) ---
    "refresh_memory.requested": {
        "ru": (
            "Хорошо, я просмотрю наши разговоры и обновлю то, что помню о тебе. "
            "Это займёт пару минут."
        ),
        "en": (
            "Sure, I'll review our conversations and update what I remember about you. "
            "This will take a couple of minutes."
        ),
    },
    "refresh_memory.in_progress": {
        "ru": "Я уже обновляю память, подожди немного.",
        "en": "I'm already refreshing my memory, give me a moment.",
    },
    "refresh_memory.enqueue_failed": {
        "ru": "Не получилось запустить обновление. Попробуй ещё раз.",
        "en": "Couldn't start the refresh. Please try again.",
    },
    # --- /settings ---
    "settings.title": {
        "ru": "Настройки",
        "en": "Settings",
    },
    "settings.choose": {
        "ru": "Что хочешь настроить?",
        "en": "What would you like to adjust?",
    },
    # --- Inline button labels ---
    "btn.tone": {"ru": "Тон", "en": "Tone"},
    "btn.persona": {"ru": "Персона", "en": "Persona"},
    "btn.skills": {"ru": "Навыки", "en": "Skills"},
    "btn.language": {"ru": "Язык", "en": "Language"},
    "btn.back": {"ru": "Назад", "en": "Back"},
    "btn.yes": {"ru": "Да", "en": "Yes"},
    "btn.no": {"ru": "Нет", "en": "No"},
    # --- Tone picker ---
    "tone.pick": {
        "ru": "Выбери тон общения:",
        "en": "Choose a communication tone:",
    },
    "tone.set": {
        "ru": "Тон изменён на «{tone}».",
        "en": "Tone set to \"{tone}\".",
    },
    # --- Confirmation (natural) ---
    "confirm.persona_change": {
        "ru": "Хочешь, чтобы я сменил персону? Подтверди, пожалуйста.",
        "en": "You'd like me to change my persona? Please confirm.",
    },
    "confirm.tone_change": {
        "ru": "Хочешь изменить тон общения? Подтверди, пожалуйста.",
        "en": "You'd like to change the tone? Please confirm.",
    },
    "confirm.skill_add_prompt": {
        "ru": "Добавить этот навык? Подтверди, пожалуйста.",
        "en": "Add this skill? Please confirm.",
    },
    "confirm.skill_remove": {
        "ru": "Убрать этот навык? Подтверди, пожалуйста.",
        "en": "Remove this skill? Please confirm.",
    },
    "confirm.generic": {
        "ru": "Применить это изменение? Подтверди, пожалуйста.",
        "en": "Apply this change? Please confirm.",
    },
    "confirm.expired": {
        "ru": "Время подтверждения истекло. Напиши запрос заново.",
        "en": "Confirmation has expired. Please send your request again.",
    },
    # --- Personas ---
    "personas.title": {
        "ru": "Выбери персону:",
        "en": "Choose a persona:",
    },
    "personas.selected": {
        "ru": "Персона «{name}» установлена!\nИзменения применятся со следующего сообщения.",
        "en": "Persona \"{name}\" set!\nChanges will take effect from your next message.",
    },
    "personas.preview": {
        "ru": "{name}\n\n{description}",
        "en": "{name}\n\n{description}",
    },
    "personas.custom_prompt": {
        "ru": "Опиши, какой персоной ты хочешь, чтобы я стал. Напиши в ответ описание.",
        "en": "Describe what persona you'd like me to be. Reply with a description.",
    },
    "btn.personas.select": {"ru": "Выбрать", "en": "Select"},
    "btn.personas.custom": {"ru": "Своя персона", "en": "Custom persona"},
    # --- Skills catalog ---
    "skills.title": {
        "ru": "Навыки",
        "en": "Skills",
    },
    "skills.description": {
        "ru": "Включай и отключай навыки. Активные отмечены галочкой.",
        "en": "Toggle skills on and off. Active ones are marked with a checkmark.",
    },
    "skills.toggled_on": {
        "ru": "Навык «{name}» включён.",
        "en": "Skill \"{name}\" enabled.",
    },
    "skills.toggled_off": {
        "ru": "Навык «{name}» отключён.",
        "en": "Skill \"{name}\" disabled.",
    },
    # --- Onboarding ---
    "onboarding.step1_name": {
        "ru": "Привет! Давай познакомимся. Как тебя зовут?",
        "en": "Hi! Let's get to know each other. What's your name?",
    },
    "onboarding.step2_interests": {
        "ru": "Приятно познакомиться, {name}! Что тебя интересует?",
        "en": "Nice to meet you, {name}! What are you interested in?",
    },
    "onboarding.step2_interests_no_name": {
        "ru": "Что тебя интересует? Это поможет мне стать полезнее.",
        "en": "What are you interested in? This helps me be more useful.",
    },
    "onboarding.step3_tone": {
        "ru": "Отлично! Как тебе удобнее общаться?",
        "en": "Great! How would you prefer to communicate?",
    },
    "onboarding.done": {
        "ru": "Готово! Я настроился под тебя. Просто напиши мне что-нибудь!",
        "en": "All set! I've tuned myself for you. Just send me a message!",
    },
    "onboarding.skip": {
        "ru": "Пропустить",
        "en": "Skip",
    },
    "onboarding.name_invalid": {
        "ru": "Имя должно быть от 1 до 64 символов без спецсимволов. Попробуй ещё раз!",
        "en": "Name should be 1\u201364 characters without special characters. Try again!",
    },
    "onboarding.please_use_buttons": {
        "ru": "Пожалуйста, выбери вариант из кнопок выше для завершения настройки.",
        "en": "Please choose an option from the buttons above to finish setup.",
    },
    # --- Interest labels for onboarding ---
    "interest.tech": {"ru": "Технологии", "en": "Technology"},
    "interest.creative": {"ru": "Творчество", "en": "Creative"},
    "interest.learning": {"ru": "Учёба", "en": "Learning"},
    "interest.fitness": {"ru": "Спорт и здоровье", "en": "Fitness & Health"},
    # --- Tone labels for onboarding ---
    "tone_label.friendly": {"ru": "Дружелюбный", "en": "Friendly"},
    "tone_label.professional": {"ru": "Деловой", "en": "Professional"},
    "tone_label.playful": {"ru": "Игривый", "en": "Playful"},
    "tone_label.concise": {"ru": "Лаконичный", "en": "Concise"},
    "tone_label.neutral": {"ru": "Нейтральный", "en": "Neutral"},
    "tone_label.casual": {"ru": "Неформальный", "en": "Casual"},
    # --- Guardrail command blocks ---
    "guardrail.command_blocked": {
        "ru": "Значение содержит недопустимый контент. Попробуй другой вариант.",
        "en": "That value contains disallowed content. Please try a different one.",
    },
    # --- Unsupported content types ---
    "unsupported.photo": {
        "ru": (
            "Я пока не умею смотреть картинки, но это в планах! "
            "Если хочешь обсудить фото — просто опиши, что на нём."
        ),
        "en": (
            "I can't see images yet, but it's on the roadmap! "
            "If you'd like to discuss a photo, just describe what's in it."
        ),
    },
    "unsupported.voice": {
        "ru": (
            "Голосовые пока не поддерживаю, но скоро научусь! "
            "А пока можешь описать словами — я внимательно прочитаю."
        ),
        "en": (
            "I can't listen to voice messages yet, but I'm learning! "
            "For now, try typing it out — I'll read every word."
        ),
    },
    "unsupported.sticker": {
        "ru": "Стикеры — это круто! Но я пока понимаю только текст. Напиши мне что-нибудь!",
        "en": "Love the sticker! But I only understand text for now. Send me a message!",
    },
    "unsupported.document": {
        "ru": (
            "Я пока не умею читать файлы. "
            "Если нужна помощь с содержимым — скопируй текст и отправь мне."
        ),
        "en": (
            "I can't read files yet. "
            "If you need help with the content, copy the text and send it to me."
        ),
    },
    "unsupported.other": {
        "ru": "Я понимаю только текстовые сообщения. Попробуй написать мне текстом!",
        "en": "I only understand text messages. Try sending me some text!",
    },
    # --- Rate limit ---
    "rate_limit.exceeded": {
        "ru": "Ты отправляешь сообщения слишком быстро. Подожди немного.",
        "en": "You're sending messages too fast. Please wait a moment.",
    },
    # --- Group chats ---
    "group.personal_command": {
        "ru": "Эта команда работает только в личных сообщениях.",
        "en": "This command is only available in private chats.",
    },
    "start.group_hint": {
        "ru": "Привет! Напиши мне в личные сообщения, чтобы настроить бота.",
        "en": "Hi! Send me a private message to set up the bot.",
    },
    # --- Continuity ---
    "prompt.continuity_instruction": {
        "ru": (
            "Пользователь возвращается после перерыва в {gap}. "
            "Поприветствуй его возвращение и естественно упомяни "
            "недавние темы: {topics}. "
            "Не перечисляй их списком — вплети в разговор."
        ),
        "en": (
            "The user is returning after being away for {gap}. "
            "Welcome them back and naturally reference "
            "these recent topics: {topics}. "
            "Don't list them — weave them into the conversation."
        ),
    },
    # --- Suggestions ---
    "prompt.suggestion_instruction": {
        "ru": (
            "На основе интересов пользователя ({interests}), "
            "предложи что-нибудь полезное или интересное в конце ответа."
        ),
        "en": (
            "Based on the user's interests ({interests}), "
            "suggest something useful or interesting at the end of your reply."
        ),
    },
}


def tr(key: str, locale: str | None = None, **kwargs: object) -> str:
    """Translate *key* for *locale* and format with ``kwargs``."""
    resolved = normalize_locale(locale)
    values = _MESSAGES[key]
    template = values.get(resolved, values["ru"])
    return template.format(**kwargs)
