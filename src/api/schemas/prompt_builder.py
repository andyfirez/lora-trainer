"""Pydantic schemas for prompt builder API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.db.tables.prompt_builder import PromptPolarity, PromptSelectionMode


class PromptSectionValueResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    section_id: int
    text: str
    sort_order: int
    created_at: datetime
    updated_at: datetime


class PromptSectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    sort_order: int
    created_at: datetime
    updated_at: datetime
    values: list[PromptSectionValueResponse] = Field(default_factory=list)


class PromptSectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PromptSectionUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PromptSectionValueCreate(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class PromptSectionValueUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class PromptTemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    polarity: PromptPolarity
    created_at: datetime
    updated_at: datetime


class PromptTemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    polarity: PromptPolarity


class PromptTemplateUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PromptTemplateSectionResponse(BaseModel):
    id: int
    template_id: int
    section_id: int
    section_name: str
    sort_order: int
    selection_mode: PromptSelectionMode
    selected_value_id: int | None
    selected_value_ids: list[int] = Field(default_factory=list)
    include_empty: bool = False
    values: list[PromptSectionValueResponse]


class PromptTemplateDetailResponse(BaseModel):
    id: int
    name: str
    polarity: PromptPolarity
    created_at: datetime
    updated_at: datetime
    sections: list[PromptTemplateSectionResponse]
    composed_prompt: str


class PromptTemplateSave(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PromptTemplateSectionCreate(BaseModel):
    section_id: int


class PromptTemplateSectionUpdate(BaseModel):
    selection_mode: PromptSelectionMode | None = None
    selected_value_id: int | None = None
    selected_value_ids: list[int] | None = None
    clear_selected_value: bool = False
    include_empty: bool | None = None


class PromptTemplateSectionReorder(BaseModel):
    order: list[int] = Field(min_length=1)
