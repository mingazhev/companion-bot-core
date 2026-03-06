"""SQLAlchemy ORM models for all database tables.

All timestamps are stored in UTC (TIMESTAMP WITH TIME ZONE).
UUIDs are used as primary keys for most tables to simplify
horizontal scaling and avoid ID leakage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Time,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------


class User(Base):
    """One row per Telegram user; internal identity anchor."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        comment="active | banned | deleted",
    )
    locale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # --- relationships ---
    profile: Mapped[UserProfile | None] = relationship(
        "UserProfile", back_populates="user", uselist=False, lazy="raise"
    )
    prompt_snapshots: Mapped[list[PromptSnapshot]] = relationship(
        "PromptSnapshot", back_populates="user", lazy="raise"
    )
    conversation_messages: Mapped[list[ConversationMessage]] = relationship(
        "ConversationMessage", back_populates="user", lazy="raise"
    )
    memory_compactions: Mapped[list[MemoryCompaction]] = relationship(
        "MemoryCompaction", back_populates="user", lazy="raise"
    )
    behavior_change_events: Mapped[list[BehaviorChangeEvent]] = relationship(
        "BehaviorChangeEvent", back_populates="user", lazy="raise"
    )
    jobs: Mapped[list[Job]] = relationship("Job", back_populates="user", lazy="raise")
    audit_log_entries: Mapped[list[AuditLog]] = relationship(
        "AuditLog", back_populates="user", lazy="raise"
    )
    conversation_sessions: Mapped[list[ConversationSession]] = relationship(
        "ConversationSession", back_populates="user", lazy="raise"
    )
    feedback_entries: Mapped[list[FeedbackEntry]] = relationship(
        "FeedbackEntry", back_populates="user", lazy="raise"
    )
    bookmarks: Mapped[list[Bookmark]] = relationship(
        "Bookmark", back_populates="user", lazy="raise"
    )
    mood_entries: Mapped[list[MoodEntry]] = relationship(
        "MoodEntry", back_populates="user", lazy="raise"
    )
    habits: Mapped[list[Habit]] = relationship(
        "Habit", back_populates="user", lazy="raise"
    )


# ---------------------------------------------------------------------------
# user_profiles
# ---------------------------------------------------------------------------


class UserProfile(Base):
    """Per-user persona configuration; updated via bot commands or refinement."""

    __tablename__ = "user_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    persona_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    tone: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="e.g. friendly, professional, playful",
    )
    style_constraints: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-text constraints on response style",
    )
    safety_level: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="moderate",
        comment="strict | moderate | relaxed",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # --- proactive messaging preferences ---
    proactive_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the user opted in to proactive check-in messages",
    )
    checkin_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
        comment="Daily check-in time in user's local timezone (HH:MM)",
    )
    quiet_hours_start: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
        comment="Start of quiet hours (no proactive messages sent)",
    )
    quiet_hours_end: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
        comment="End of quiet hours",
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="profile")


# ---------------------------------------------------------------------------
# prompt_snapshots
# ---------------------------------------------------------------------------


class PromptSnapshot(Base):
    """Immutable versioned snapshot of a user's full system prompt state."""

    __tablename__ = "prompt_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    skill_prompts_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Map of skill_name -> skill_system_prompt",
    )
    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="initial | user_command | refinement | rollback | behavior_change",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="prompt_snapshots")


# ---------------------------------------------------------------------------
# conversation_messages
# ---------------------------------------------------------------------------


class ConversationMessage(Base):
    """A single message in the user's conversation history.

    Rows with ``ttl_expires_at`` in the past are eligible for deletion by the
    privacy TTL sweeper job.
    """

    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index(
            "ix_conversation_messages_user_id_created_at",
            "user_id",
            "created_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="user | assistant | system",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    ttl_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Row is eligible for deletion after this timestamp",
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="conversation_messages")


# ---------------------------------------------------------------------------
# memory_compactions
# ---------------------------------------------------------------------------


class MemoryCompaction(Base):
    """Record of a compaction (summarisation) pass on a user's conversation history."""

    __tablename__ = "memory_compactions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    snapshot_id_before: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("prompt_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    snapshot_id_after: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("prompt_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="pending | done | failed",
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="memory_compactions")


# ---------------------------------------------------------------------------
# behavior_change_events
# ---------------------------------------------------------------------------


class BehaviorChangeEvent(Base):
    """Detector output: a potential user-driven configuration change."""

    __tablename__ = "behavior_change_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    intent: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="tone_change | persona_change | skill_add_prompt | skill_remove | "
        "safety_override_attempt | normal_chat",
    )
    risk_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="low | medium | high",
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="For medium-risk: did the user confirm the change?",
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="UUID of the ConversationMessage that triggered detection (soft ref)",
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="behavior_change_events")


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


class Job(Base):
    """Async background job (refinement, compaction, etc.)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="refinement | compaction | ttl_sweep",
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="pending | running | done | skipped | failed | dead_letter",
    )
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="jobs")


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------


class AuditLog(Base):
    """Append-only audit trail; preserved by the hard-delete flow."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Nullable: system events may not be tied to a user",
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    details_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Minimal details; must not contain PII",
    )

    # --- relationships ---
    user: Mapped[User | None] = relationship("User", back_populates="audit_log_entries")


# ---------------------------------------------------------------------------
# conversation_sessions
# ---------------------------------------------------------------------------


class ConversationSession(Base):
    """Tracks a contiguous block of user messages as one session.

    A new session starts when the gap since the last message exceeds 30 minutes.
    """

    __tablename__ = "conversation_sessions"
    __table_args__ = (
        Index(
            "ix_conversation_sessions_user_id_started_at",
            "user_id",
            "started_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    ended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    dominant_mood: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Most recent non-neutral emotion mode detected during session",
    )
    ended_with_farewell: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="conversation_sessions")


# ---------------------------------------------------------------------------
# feedback_entries
# ---------------------------------------------------------------------------


class FeedbackEntry(Base):
    """User satisfaction feedback collected naturally within conversation.

    Triggered after every N-th session at farewell, with a once-per-week
    per-user cooldown.
    """

    __tablename__ = "feedback_entries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        nullable=True,
        comment="Session during which feedback was collected",
    )
    raw_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original user response to the feedback prompt",
    )
    sentiment_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Classified sentiment 1 (negative) to 5 (positive)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="feedback_entries")


# ---------------------------------------------------------------------------
# bookmarks
# ---------------------------------------------------------------------------


class Bookmark(Base):
    """A saved conversation moment (user message + bot response pair).

    Users trigger bookmarks via natural language ("запомни это", "сохрани")
    or browse them with the /bookmarks command.
    """

    __tablename__ = "bookmarks"
    __table_args__ = (
        Index(
            "ix_bookmarks_user_id_created_at",
            "user_id",
            "created_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The user message that was bookmarked",
    )
    bot_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The bot response paired with the user message",
    )
    tag: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Optional user-specified tag for categorisation",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="bookmarks")


# ---------------------------------------------------------------------------
# mood_entries
# ---------------------------------------------------------------------------


class MoodEntry(Base):
    """Automatic mood tracking entry derived from emotion detection.

    Created after the emotion detector runs on each message when the
    detected mode is non-neutral.
    """

    __tablename__ = "mood_entries"
    __table_args__ = (
        Index(
            "ix_mood_entries_user_id_created_at",
            "user_id",
            "created_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    mood: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="happy | sad | anxious | angry | neutral | excited",
    )
    intensity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Intensity 1 (low) to 5 (high)",
    )
    context_snippet: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="First 50 chars of the user message (encrypted at rest when enabled)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="mood_entries")


# ---------------------------------------------------------------------------
# habits
# ---------------------------------------------------------------------------


class Habit(Base):
    """A user-defined habit with streak tracking.

    Users create habits via natural language ("хочу каждый день читать")
    or browse them with the /habits command.  Streak resets silently on miss.
    """

    __tablename__ = "habits"
    __table_args__ = (
        Index(
            "ix_habits_user_id_created_at",
            "user_id",
            "created_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Human-readable habit title",
    )
    frequency: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="daily",
        comment="daily | weekly",
    )
    current_streak: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Current consecutive completion streak",
    )
    best_streak: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="All-time best streak",
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of last check-in for this habit",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set when the user archives (soft-deletes) the habit",
    )

    # --- relationships ---
    user: Mapped[User] = relationship("User", back_populates="habits")
