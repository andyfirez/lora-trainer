"""Business logic for prompt builder CRUD and composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.db.repositories.prompt_builder_repo import (
    PromptSectionRepository,
    PromptSectionValueRepository,
    PromptTemplateRepository,
    PromptTemplateSectionRepository,
)
from src.db.tables.prompt_builder import (
    PromptPolarity,
    PromptSection,
    PromptSectionValue,
    PromptSelectionMode,
    PromptTemplate,
    PromptTemplateSection,
)
from src.services.prompt_builder.compose import (
    compose_template_prompt,
    validate_section_value_text,
)
from src.services.prompt_builder.exceptions import (
    InvalidPromptValueTextError,
    PromptSectionAlreadyLinkedError,
    PromptSectionInUseError,
    PromptSectionNotFoundError,
    PromptSectionValueInUseError,
    PromptSectionValueNotFoundError,
    PromptTemplateNotFoundError,
    PromptTemplatePolarityMismatchError,
    PromptTemplateSectionNotFoundError,
    PromptWorkspaceTemplateError,
)

_WORKSPACE_LABELS = {
    PromptPolarity.POSITIVE: "Current (positive)",
    PromptPolarity.NEGATIVE: "Current (negative)",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PromptSectionDetail:
    section: PromptSection
    values: list[PromptSectionValue]


@dataclass(frozen=True)
class PromptTemplateSectionDetail:
    link: PromptTemplateSection
    section: PromptSection
    values: list[PromptSectionValue]


@dataclass(frozen=True)
class PromptTemplateDetail:
    template: PromptTemplate
    sections: list[PromptTemplateSectionDetail]
    composed_prompt: str


class PromptBuilderService:
    def __init__(
        self,
        section_repo: PromptSectionRepository,
        value_repo: PromptSectionValueRepository,
        template_repo: PromptTemplateRepository,
        template_section_repo: PromptTemplateSectionRepository,
    ) -> None:
        self._section_repo = section_repo
        self._value_repo = value_repo
        self._template_repo = template_repo
        self._template_section_repo = template_section_repo

    async def list_sections(self) -> list[PromptSectionDetail]:
        sections = await self._section_repo.list_with_values()
        details: list[PromptSectionDetail] = []
        for section in sections:
            values = list(await self._value_repo.list_for_section(section.id))  # type: ignore[arg-type]
            details.append(PromptSectionDetail(section=section, values=values))
        return details

    async def create_section(self, *, name: str) -> PromptSectionDetail:
        sort_order = await self._template_section_repo.next_section_sort_order()
        section = PromptSection(name=name.strip(), sort_order=sort_order)
        section = await self._section_repo.add(section)
        return PromptSectionDetail(section=section, values=[])

    async def update_section(self, section_id: int, *, name: str) -> PromptSection:
        section = await self._get_section(section_id)
        section.name = name.strip()
        section.updated_at = _utcnow()
        return await self._section_repo.save_flush_refresh(section)

    async def delete_section(self, section_id: int) -> None:
        section = await self._get_section(section_id)
        if await self._section_repo.count_template_references(section_id) > 0:
            raise PromptSectionInUseError(section_id)
        values = await self._value_repo.list_for_section(section_id)
        for value in values:
            await self._value_repo.delete(value)
        await self._section_repo.delete(section)

    async def list_section_values(self, section_id: int) -> list[PromptSectionValue]:
        await self._get_section(section_id)
        return list(await self._value_repo.list_for_section(section_id))

    async def create_section_value(self, section_id: int, *, text: str) -> PromptSectionValue:
        await self._get_section(section_id)
        cleaned = text.strip()
        self._validate_value_text(cleaned)
        sort_order = await self._template_section_repo.next_value_sort_order(section_id)
        value = PromptSectionValue(section_id=section_id, text=cleaned, sort_order=sort_order)
        return await self._value_repo.add(value)

    async def update_section_value(self, value_id: int, *, text: str) -> PromptSectionValue:
        value = await self._get_value(value_id)
        cleaned = text.strip()
        self._validate_value_text(cleaned)
        value.text = cleaned
        value.updated_at = _utcnow()
        return await self._value_repo.save_flush_refresh(value)

    async def delete_section_value(self, value_id: int) -> None:
        value = await self._get_value(value_id)
        if await self._value_repo.count_selection_references(value_id) > 0:
            raise PromptSectionValueInUseError(value_id)
        await self._value_repo.delete(value)

    async def list_templates(self, polarity: PromptPolarity | None = None) -> list[PromptTemplate]:
        return list(await self._template_repo.list_saved_templates(polarity))

    async def get_workspace(self, polarity: PromptPolarity) -> PromptTemplateDetail:
        workspace = await self._get_or_create_workspace(polarity)
        return await self._build_template_detail(workspace)

    async def save_workspace_as_template(self, *, polarity: PromptPolarity, name: str) -> PromptTemplate:
        workspace = await self._get_or_create_workspace(polarity)
        template = PromptTemplate(
            name=name.strip(),
            polarity=polarity,
            is_workspace=False,
        )
        template = await self._template_repo.add(template)
        await self._copy_template_sections(workspace.id, template.id)  # type: ignore[arg-type]
        return template

    async def load_template_into_workspace(self, template_id: int) -> PromptTemplateDetail:
        template = await self._get_template(template_id)
        if template.is_workspace:
            raise PromptWorkspaceTemplateError("Cannot load a workspace template")
        workspace = await self._get_or_create_workspace(template.polarity)
        if template.polarity != workspace.polarity:
            raise PromptTemplatePolarityMismatchError(
                template_id,
                expected=workspace.polarity,
                actual=template.polarity,
            )
        await self._replace_template_sections(template.id, workspace.id)  # type: ignore[arg-type]
        return await self._build_template_detail(workspace)

    async def create_template(self, *, name: str, polarity: PromptPolarity) -> PromptTemplate:
        template = PromptTemplate(name=name.strip(), polarity=polarity, is_workspace=False)
        return await self._template_repo.add(template)

    async def update_template(self, template_id: int, *, name: str) -> PromptTemplate:
        template = await self._get_template(template_id)
        template.name = name.strip()
        template.updated_at = _utcnow()
        return await self._template_repo.save_flush_refresh(template)

    async def delete_template(self, template_id: int) -> None:
        template = await self._get_template(template_id)
        if template.is_workspace:
            raise PromptWorkspaceTemplateError("Workspace cannot be deleted")
        links = await self._template_section_repo.list_for_template(template_id)
        for link in links:
            await self._template_section_repo.delete(link)
        await self._template_repo.delete(template)

    async def add_workspace_section(self, polarity: PromptPolarity, *, section_id: int) -> PromptTemplateDetail:
        workspace = await self._get_or_create_workspace(polarity)
        return await self.add_template_section(workspace.id, section_id=section_id)  # type: ignore[arg-type]

    async def update_workspace_section(
        self,
        polarity: PromptPolarity,
        link_id: int,
        *,
        selection_mode: PromptSelectionMode | None = None,
        selected_value_id: int | None = None,
        selected_value_ids: list[int] | None = None,
        clear_selected_value: bool = False,
        include_empty: bool | None = None,
    ) -> PromptTemplateDetail:
        workspace = await self._get_or_create_workspace(polarity)
        return await self.update_template_section(
            workspace.id,  # type: ignore[arg-type]
            link_id,
            selection_mode=selection_mode,
            selected_value_id=selected_value_id,
            selected_value_ids=selected_value_ids,
            clear_selected_value=clear_selected_value,
            include_empty=include_empty,
        )

    async def reorder_workspace_sections(
        self,
        polarity: PromptPolarity,
        *,
        order: list[int],
    ) -> PromptTemplateDetail:
        workspace = await self._get_or_create_workspace(polarity)
        return await self.reorder_template_sections(workspace.id, order=order)  # type: ignore[arg-type]

    async def remove_workspace_section(self, polarity: PromptPolarity, link_id: int) -> None:
        workspace = await self._get_or_create_workspace(polarity)
        await self.remove_template_section(workspace.id, link_id)  # type: ignore[arg-type]

    async def get_template_detail(self, template_id: int) -> PromptTemplateDetail:
        template = await self._get_template(template_id)
        return await self._build_template_detail(template)

    async def add_template_section(self, template_id: int, *, section_id: int) -> PromptTemplateDetail:
        await self._get_template(template_id)
        await self._get_section(section_id)
        if await self._template_section_repo.section_already_in_template(template_id, section_id):
            raise PromptSectionAlreadyLinkedError(section_id, template_id)

        values = list(await self._value_repo.list_for_section(section_id))
        selected_ids = [values[0].id] if values else []
        sort_order = await self._template_section_repo.next_sort_order(template_id)
        link = PromptTemplateSection(
            template_id=template_id,
            section_id=section_id,
            sort_order=sort_order,
            selection_mode=PromptSelectionMode.FIXED,
            selected_value_id=selected_ids[0] if selected_ids else None,
            selected_value_ids=selected_ids,
        )
        await self._template_section_repo.add(link)
        template = await self._get_template(template_id)
        return await self._build_template_detail(template)

    async def update_template_section(
        self,
        template_id: int,
        link_id: int,
        *,
        selection_mode: PromptSelectionMode | None = None,
        selected_value_id: int | None = None,
        selected_value_ids: list[int] | None = None,
        clear_selected_value: bool = False,
        include_empty: bool | None = None,
    ) -> PromptTemplateDetail:
        link = await self._get_template_link(template_id, link_id)
        if selection_mode is not None:
            link.selection_mode = selection_mode
        if include_empty is not None:
            link.include_empty = include_empty
        if clear_selected_value:
            link.selected_value_id = None
            link.selected_value_ids = []
        elif selected_value_ids is not None:
            normalized = await self._validate_selected_value_ids(link.section_id, selected_value_ids)
            link.selected_value_ids = normalized
            link.selected_value_id = normalized[0] if normalized else None
        elif selected_value_id is not None:
            value = await self._get_value(selected_value_id)
            if value.section_id != link.section_id:
                raise InvalidPromptValueTextError(
                    f"Value {selected_value_id} does not belong to section {link.section_id}"
                )
            link.selected_value_id = selected_value_id
            link.selected_value_ids = [selected_value_id]
        link.updated_at = _utcnow()
        await self._template_section_repo.save_flush_refresh(link)
        template = await self._get_template(template_id)
        return await self._build_template_detail(template)

    async def _validate_selected_value_ids(
        self,
        section_id: int,
        selected_value_ids: list[int],
    ) -> list[int]:
        if not selected_value_ids:
            return []
        seen: set[int] = set()
        normalized: list[int] = []
        for value_id in selected_value_ids:
            if value_id in seen:
                continue
            value = await self._get_value(value_id)
            if value.section_id != section_id:
                raise InvalidPromptValueTextError(
                    f"Value {value_id} does not belong to section {section_id}"
                )
            seen.add(value_id)
            normalized.append(value_id)
        section_values = list(await self._value_repo.list_for_section(section_id))
        order = {value.id: index for index, value in enumerate(section_values)}
        return sorted(normalized, key=lambda vid: order.get(vid, vid))

    async def reorder_template_sections(
        self,
        template_id: int,
        *,
        order: list[int],
    ) -> PromptTemplateDetail:
        await self._get_template(template_id)
        links = list(await self._template_section_repo.list_for_template(template_id))
        link_ids = {link.id for link in links}
        if set(order) != link_ids:
            raise InvalidPromptValueTextError("Reorder order must include every template section link id exactly once")
        link_by_id = {link.id: link for link in links}
        for index, link_id in enumerate(order):
            link = link_by_id[link_id]
            link.sort_order = index
            link.updated_at = _utcnow()
            self._template_section_repo.save(link)
        await self._template_section_repo.flush()
        template = await self._get_template(template_id)
        return await self._build_template_detail(template)

    async def remove_template_section(self, template_id: int, link_id: int) -> None:
        link = await self._get_template_link(template_id, link_id)
        await self._template_section_repo.delete(link)

    async def _build_template_detail(self, template: PromptTemplate) -> PromptTemplateDetail:
        links = list(await self._template_section_repo.list_for_template(template.id))  # type: ignore[arg-type]
        section_details: list[PromptTemplateSectionDetail] = []
        values_by_section: dict[int, list[PromptSectionValue]] = {}
        for link in links:
            section = await self._get_section(link.section_id)
            values = list(await self._value_repo.list_for_section(link.section_id))
            values_by_section[link.section_id] = values
            section_details.append(
                PromptTemplateSectionDetail(link=link, section=section, values=values)
            )
        composed = compose_template_prompt(links, values_by_section)
        return PromptTemplateDetail(
            template=template,
            sections=section_details,
            composed_prompt=composed,
        )

    async def _get_or_create_workspace(self, polarity: PromptPolarity) -> PromptTemplate:
        workspace = await self._template_repo.get_workspace(polarity)
        if workspace is not None:
            return workspace
        workspace = PromptTemplate(
            name=_WORKSPACE_LABELS[polarity],
            polarity=polarity,
            is_workspace=True,
        )
        return await self._template_repo.add(workspace)

    async def _copy_template_sections(self, source_template_id: int, dest_template_id: int) -> None:
        links = list(await self._template_section_repo.list_for_template(source_template_id))
        for link in links:
            copy = PromptTemplateSection(
                template_id=dest_template_id,
                section_id=link.section_id,
                sort_order=link.sort_order,
                selection_mode=link.selection_mode,
                selected_value_id=link.selected_value_id,
                selected_value_ids=list(link.selected_value_ids or []),
                include_empty=link.include_empty,
            )
            await self._template_section_repo.add(copy)

    async def _replace_template_sections(self, source_template_id: int, dest_template_id: int) -> None:
        existing = list(await self._template_section_repo.list_for_template(dest_template_id))
        for link in existing:
            await self._template_section_repo.delete(link)
        await self._copy_template_sections(source_template_id, dest_template_id)

    async def _get_section(self, section_id: int) -> PromptSection:
        section = await self._section_repo.get_by_id(section_id)
        if section is None:
            raise PromptSectionNotFoundError(section_id)
        return section

    async def _get_value(self, value_id: int) -> PromptSectionValue:
        value = await self._value_repo.get_by_id(value_id)
        if value is None:
            raise PromptSectionValueNotFoundError(value_id)
        return value

    async def _get_template(self, template_id: int) -> PromptTemplate:
        template = await self._template_repo.get_by_id(template_id)
        if template is None:
            raise PromptTemplateNotFoundError(template_id)
        return template

    async def _get_template_link(self, template_id: int, link_id: int) -> PromptTemplateSection:
        link = await self._template_section_repo.get_for_template(template_id, link_id)
        if link is None:
            raise PromptTemplateSectionNotFoundError(link_id)
        return link

    @staticmethod
    def _validate_value_text(text: str) -> None:
        try:
            validate_section_value_text(text)
        except ValueError as exc:
            raise InvalidPromptValueTextError(str(exc)) from exc
