"""Repository layer for prompt builder tables."""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.prompt_builder import (
    PromptPolarity,
    PromptSection,
    PromptSectionValue,
    PromptTemplate,
    PromptTemplateSection,
)
from src.services.prompt_builder.compose import resolve_selected_value_ids


class PromptSectionRepository(BaseRepository[PromptSection]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(PromptSection, session)

    async def list_with_values(self) -> Sequence[PromptSection]:
        return await self.list_ordered(PromptSection.sort_order, PromptSection.id)

    async def count_template_references(self, section_id: int) -> int:
        statement = select(func.count()).select_from(PromptTemplateSection).where(
            PromptTemplateSection.section_id == section_id
        )
        result = await self._exec(statement)
        return int(result.scalar_one())


class PromptSectionValueRepository(BaseRepository[PromptSectionValue]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(PromptSectionValue, session)

    async def list_for_section(self, section_id: int) -> Sequence[PromptSectionValue]:
        statement = (
            select(PromptSectionValue)
            .where(PromptSectionValue.section_id == section_id)
            .order_by(PromptSectionValue.sort_order, PromptSectionValue.id)
        )
        result = await self._exec(statement)
        return result.scalars().all()

    async def count_selection_references(self, value_id: int) -> int:
        result = await self._exec(select(PromptTemplateSection))
        return sum(
            1
            for link in result.scalars().all()
            if value_id in resolve_selected_value_ids(link)
        )


class PromptTemplateRepository(BaseRepository[PromptTemplate]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(PromptTemplate, session)

    async def list_saved_templates(self, polarity: PromptPolarity | None = None) -> Sequence[PromptTemplate]:
        statement = select(PromptTemplate).where(PromptTemplate.is_workspace.is_(False))
        if polarity is not None:
            statement = statement.where(PromptTemplate.polarity == polarity)
        statement = statement.order_by(PromptTemplate.name, PromptTemplate.id)
        result = await self._exec(statement)
        return result.scalars().all()

    async def get_workspace(self, polarity: PromptPolarity) -> PromptTemplate | None:
        statement = select(PromptTemplate).where(
            PromptTemplate.polarity == polarity,
            PromptTemplate.is_workspace.is_(True),
        )
        result = await self._exec(statement)
        return result.scalars().first()


class PromptTemplateSectionRepository(BaseRepository[PromptTemplateSection]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(PromptTemplateSection, session)

    async def list_for_template(self, template_id: int) -> Sequence[PromptTemplateSection]:
        statement = (
            select(PromptTemplateSection)
            .where(PromptTemplateSection.template_id == template_id)
            .order_by(PromptTemplateSection.sort_order, PromptTemplateSection.id)
        )
        result = await self._exec(statement)
        return result.scalars().all()

    async def get_for_template(self, template_id: int, link_id: int) -> PromptTemplateSection | None:
        statement = select(PromptTemplateSection).where(
            PromptTemplateSection.id == link_id,
            PromptTemplateSection.template_id == template_id,
        )
        result = await self._exec(statement)
        return result.scalars().first()

    async def section_already_in_template(self, template_id: int, section_id: int) -> bool:
        statement = select(PromptTemplateSection).where(
            PromptTemplateSection.template_id == template_id,
            PromptTemplateSection.section_id == section_id,
        )
        result = await self._exec(statement)
        return result.scalars().first() is not None

    async def next_sort_order(self, template_id: int) -> int:
        statement = select(func.max(PromptTemplateSection.sort_order)).where(
            PromptTemplateSection.template_id == template_id
        )
        result = await self._exec(statement)
        current = result.scalar_one()
        return (current if current is not None else -1) + 1

    async def next_section_sort_order(self) -> int:
        statement = select(func.max(PromptSection.sort_order))
        result = await self._exec(statement)
        current = result.scalar_one()
        return (current if current is not None else -1) + 1

    async def next_value_sort_order(self, section_id: int) -> int:
        statement = select(func.max(PromptSectionValue.sort_order)).where(
            PromptSectionValue.section_id == section_id
        )
        result = await self._exec(statement)
        current = result.scalar_one()
        return (current if current is not None else -1) + 1
