"""Integration tests for prompt builder service."""

import pytest
import pytest_asyncio
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.prompt_builder_repo import (
    PromptSectionRepository,
    PromptSectionValueRepository,
    PromptTemplateRepository,
    PromptTemplateSectionRepository,
)
from src.db.tables.prompt_builder import PromptPolarity
from src.services.prompt_builder.exceptions import PromptSectionInUseError
from src.services.prompt_builder.service import PromptBuilderService


@pytest_asyncio.fixture
async def prompt_builder_service(session: AsyncSession) -> PromptBuilderService:
    return PromptBuilderService(
        PromptSectionRepository(session),
        PromptSectionValueRepository(session),
        PromptTemplateRepository(session),
        PromptTemplateSectionRepository(session),
    )


@pytest.mark.asyncio
async def test_workspace_crud_save_and_load(prompt_builder_service: PromptBuilderService) -> None:
    section = await prompt_builder_service.create_section(name="Subject")
    value_a = await prompt_builder_service.create_section_value(section.section.id, text="1girl")
    await prompt_builder_service.create_section_value(section.section.id, text="2girls")

    style = await prompt_builder_service.create_section(name="Style")
    await prompt_builder_service.create_section_value(style.section.id, text="anime")
    await prompt_builder_service.create_section_value(style.section.id, text="realistic")

    await prompt_builder_service.get_workspace(PromptPolarity.POSITIVE)
    await prompt_builder_service.add_workspace_section(
        PromptPolarity.POSITIVE,
        section_id=section.section.id,
    )
    detail = await prompt_builder_service.add_workspace_section(
        PromptPolarity.POSITIVE,
        section_id=style.section.id,
    )

    subject_link = next(entry for entry in detail.sections if entry.section.name == "Subject")
    detail = await prompt_builder_service.update_workspace_section(
        PromptPolarity.POSITIVE,
        subject_link.link.id,  # type: ignore[arg-type]
        selected_value_ids=[value_a.id],  # type: ignore[list-item]
    )
    style_link = next(entry for entry in detail.sections if entry.section.name == "Style")
    style_values = [value.id for value in style_link.values]
    detail = await prompt_builder_service.update_workspace_section(
        PromptPolarity.POSITIVE,
        style_link.link.id,  # type: ignore[arg-type]
        selected_value_ids=style_values,
    )

    assert detail.composed_prompt == "1girl, anime, realistic"

    saved = await prompt_builder_service.save_workspace_as_template(
        polarity=PromptPolarity.POSITIVE,
        name="Portrait",
    )
    assert saved.is_workspace is False

    await prompt_builder_service.remove_workspace_section(
        PromptPolarity.POSITIVE,
        detail.sections[0].link.id,  # type: ignore[arg-type]
    )
    await prompt_builder_service.remove_workspace_section(
        PromptPolarity.POSITIVE,
        detail.sections[1].link.id,  # type: ignore[arg-type]
    )
    empty = await prompt_builder_service.get_workspace(PromptPolarity.POSITIVE)
    assert empty.composed_prompt == ""

    loaded = await prompt_builder_service.load_template_into_workspace(saved.id)  # type: ignore[arg-type]
    assert loaded.composed_prompt == "1girl, anime, realistic"


@pytest.mark.asyncio
async def test_delete_section_in_use_raises(prompt_builder_service: PromptBuilderService) -> None:
    section = await prompt_builder_service.create_section(name="Locked")
    await prompt_builder_service.get_workspace(PromptPolarity.NEGATIVE)
    await prompt_builder_service.add_workspace_section(
        PromptPolarity.NEGATIVE,
        section_id=section.section.id,
    )

    with pytest.raises(PromptSectionInUseError):
        await prompt_builder_service.delete_section(section.section.id)
