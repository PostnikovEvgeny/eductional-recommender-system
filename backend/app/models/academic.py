from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base


class DirectionCore(Base):
    """Образовательное ядро ПНИПУ (например, 'Математическое и IT ядро')."""

    __tablename__ = "direction_cores"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Связь с направлениями подготовки
    directions: Mapped[list["StudyDirection"]] = relationship(
        back_populates="core", cascade="all, delete-orphan"
    )


class Faculty(Base):
    """Факультет университета (например, 'Электротехнический факультет')."""

    __tablename__ = "faculties"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    directions: Mapped[list["StudyDirection"]] = relationship(
        back_populates="faculty", cascade="all, delete-orphan"
    )


class StudyDirection(Base):
    """Направление подготовки / образовательная программа с интегральным эмбеддингом."""

    __tablename__ = "study_directions"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    faculty_id: Mapped[int] = mapped_column(
        ForeignKey("faculties.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    core_id: Mapped[int] = mapped_column(
        ForeignKey("direction_cores.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )

    # Связи
    faculty: Mapped["Faculty"] = relationship(back_populates="directions")
    core: Mapped["DirectionCore"] = relationship(back_populates="directions")
    disciplines: Mapped[list["Discipline"]] = relationship(
        back_populates="direction", cascade="all, delete-orphan"
    )
    domain_affiliations: Mapped[list["DirectionDomainAffiliation"]] = relationship(
        back_populates="direction", cascade="all, delete-orphan"
    )
    admission_conditions: Mapped[list["AdmissionCondition"]] = relationship(
        back_populates="direction", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_study_directions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Discipline(Base):
    """Профильная (вариативная) дисциплина учебного плана с текстом аннотации."""

    __tablename__ = "disciplines"

    id: Mapped[int] = mapped_column(primary_key=True)
    direction_id: Mapped[int] = mapped_column(
        ForeignKey("study_directions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    annotation_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )

    direction: Mapped["StudyDirection"] = relationship(back_populates="disciplines")