from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import SemanticMemory


class MemoryService:
    def __init__(self, db: Session):
        self.db = db

    def store_reference(
        self,
        key: str,
        value: str,
        camera_id: str | None = None,
        region: str | None = None,
        metadata: dict | None = None,
    ) -> SemanticMemory:
        existing = self.db.scalar(
            select(SemanticMemory).where(SemanticMemory.key == key)
        )

        if existing:
            existing.value = value
            existing.camera_id = camera_id
            existing.region = region
            existing.metadata_ = metadata or {}
            self.db.commit()
            self.db.refresh(existing)
            return existing

        memory = SemanticMemory(
            key=key,
            value=value,
            camera_id=camera_id,
            region=region,
            metadata_=metadata or {},
        )

        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)

        return memory

    def resolve_reference(self, key: str) -> SemanticMemory | None:
        return self.db.scalar(
            select(SemanticMemory).where(SemanticMemory.key == key)
        )