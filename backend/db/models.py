from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )


class Object(Base):
    __tablename__ = "objects"

    object_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )

    object_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


class Observation(Base):
    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )

    camera_id: Mapped[str] = mapped_column(
        ForeignKey("cameras.camera_id"),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        nullable=False,
    )

    frame_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    object_id: Mapped[str | None] = mapped_column(
        ForeignKey("objects.object_id"),
        nullable=True,
    )

    object_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    bbox: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    frame_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    crop_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_video: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class Event(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )

    camera_id: Mapped[str] = mapped_column(
        ForeignKey("cameras.camera_id"),
        nullable=False,
    )

    object_id: Mapped[str | None] = mapped_column(
        ForeignKey("objects.object_id"),
        nullable=True,
    )

    timestamp_start: Mapped[datetime] = mapped_column(
        nullable=False,
    )

    timestamp_end: Mapped[datetime] = mapped_column(
        nullable=False,
    )

    best_timestamp: Mapped[datetime] = mapped_column(
        nullable=False,
    )

    score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    object_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    source_video: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class SemanticMemory(Base):
    __tablename__ = "semantic_memories"

    memory_id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    key: Mapped[str] = mapped_column(
        String(200),
        unique=True,
        nullable=False,
    )

    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    camera_id: Mapped[str | None] = mapped_column(
        ForeignKey("cameras.camera_id"),
        nullable=True,
    )

    region: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Query(Base):
    __tablename__ = "queries"

    query_id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
    )

    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
    )