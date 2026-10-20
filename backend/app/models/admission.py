from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Campus(Base):
    """Филиал университета (например, 'Главный корпус г. Перми', 'Березниковский филиал')."""

    __tablename__ = "campuses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    admission_conditions: Mapped[list["AdmissionCondition"]] = relationship(
        back_populates="campus", cascade="all, delete-orphan"
    )


class ExamSubject(Base):
    """Общеобразовательный предмет ЕГЭ (например, 'Математика', 'Информатика и ИКТ')."""

    __tablename__ = "exam_subjects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)

    exam_requirements: Mapped[list["AdmissionExamRequirement"]] = relationship(
        back_populates="exam_subject", cascade="all, delete-orphan"
    )


class AdmissionCondition(Base):
    """Условия приема на направление подготовки (форма, основа, филиал, проходные баллы)."""

    __tablename__ = "admission_conditions"

    id: Mapped[int] = mapped_column(primary_key=True)
    direction_id: Mapped[int] = mapped_column(
        ForeignKey("study_directions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    campus_id: Mapped[int] = mapped_column(
        ForeignKey("campuses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    study_form: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    funding_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    is_foreign_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    passing_score_prev_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    budget_places: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    direction: Mapped["StudyDirection"] = relationship(back_populates="admission_conditions")
    campus: Mapped["Campus"] = relationship(back_populates="admission_conditions")
    exam_requirements: Mapped[list["AdmissionExamRequirement"]] = relationship(
        back_populates="condition", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "study_form IN ('очная', 'заочная', 'очно-заочная')",
            name="study_form_allowed",
        ),
        CheckConstraint(
            "funding_type IN ('бюджет', 'контракт')",
            name="funding_type_allowed",
        ),
        UniqueConstraint(
            "direction_id",
            "campus_id",
            "study_form",
            "funding_type",
            name="uq_admission_condition_variant",
        ),
    )


class AdmissionExamRequirement(Base):
    """Требуемый предмет ЕГЭ и минимальный пороговый балл для конкретного условия приема."""

    __tablename__ = "admission_exam_requirements"

    id: Mapped[int] = mapped_column(primary_key=True)
    condition_id: Mapped[int] = mapped_column(
        ForeignKey("admission_conditions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    exam_id: Mapped[int] = mapped_column(
        ForeignKey("exam_subjects.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    min_score: Mapped[int] = mapped_column(Integer, nullable=False)

    condition: Mapped["AdmissionCondition"] = relationship(back_populates="exam_requirements")
    exam_subject: Mapped["ExamSubject"] = relationship(back_populates="exam_requirements")

    __table_args__ = (
        CheckConstraint(
            "min_score >= 0 AND min_score <= 100",
            name="min_score_range",
        ),
        UniqueConstraint("condition_id", "exam_id", name="uq_condition_exam"),
    )