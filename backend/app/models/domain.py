from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base


class Domain(Base):
    """Предметный домен (область интереса абитуриента) с уровнем семантического сжатия."""

    __tablename__ = "domains"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    compression_level: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )

    direction_affiliations: Mapped[list["DirectionDomainAffiliation"]] = relationship(
        back_populates="domain", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "compression_level IN (1, 3, 5)",
            name="compression_level_allowed_values",
        ),
        UniqueConstraint("name", "compression_level", name="uq_domains_name_level"),
        Index(
            "ix_domains_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class DirectionDomainAffiliation(Base):
    """Степень принадлежности (косинусной близости μ) домена к направлению подготовки."""

    __tablename__ = "direction_domain_affiliations"

    id: Mapped[int] = mapped_column(primary_key=True)
    direction_id: Mapped[int] = mapped_column(
        ForeignKey("study_directions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    domain_id: Mapped[int] = mapped_column(
        ForeignKey("domains.id", ondelete="CASCADE"), nullable=False, index=True
    )
    membership_degree: Mapped[float] = mapped_column(Float, nullable=False)

    direction: Mapped["StudyDirection"] = relationship(back_populates="domain_affiliations")
    domain: Mapped["Domain"] = relationship(back_populates="direction_affiliations")

    __table_args__ = (
        CheckConstraint(
            "membership_degree >= 0.0 AND membership_degree <= 1.0",
            name="membership_degree_range",
        ),
        UniqueConstraint(
            "direction_id", "domain_id", name="uq_affiliation_direction_domain"
        ),
    )