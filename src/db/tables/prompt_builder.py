"""Prompt builder SQLModel tables."""

from enum import StrEnum
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class PromptPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class PromptSelectionMode(StrEnum):
    FIXED = "fixed"
    RANDOM = "random"


class PromptSection(TimestampMixin, SQLModel, table=True):
    __tablename__ = "prompt_sections"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    sort_order: int = Field(default=0, index=True)


class PromptSectionValue(TimestampMixin, SQLModel, table=True):
    __tablename__ = "prompt_section_values"

    id: Optional[int] = Field(default=None, primary_key=True)
    section_id: int = Field(foreign_key="prompt_sections.id", index=True)
    text: str
    sort_order: int = Field(default=0, index=True)


class PromptTemplate(TimestampMixin, SQLModel, table=True):
    __tablename__ = "prompt_templates"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    polarity: PromptPolarity = Field(
        sa_column=sa.Column(sa.String(), nullable=False, index=True),
        default=PromptPolarity.POSITIVE,
    )
    is_workspace: bool = Field(default=False, index=True)


class PromptTemplateSection(TimestampMixin, SQLModel, table=True):
    __tablename__ = "prompt_template_sections"

    id: Optional[int] = Field(default=None, primary_key=True)
    template_id: int = Field(foreign_key="prompt_templates.id", index=True)
    section_id: int = Field(foreign_key="prompt_sections.id", index=True)
    sort_order: int = Field(default=0, index=True)
    selection_mode: PromptSelectionMode = Field(
        sa_column=sa.Column(sa.String(), nullable=False),
        default=PromptSelectionMode.FIXED,
    )
    selected_value_id: Optional[int] = Field(
        default=None,
        foreign_key="prompt_section_values.id",
    )
    selected_value_ids: Optional[list[int]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    include_empty: bool = Field(default=False)
