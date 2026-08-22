"""Prompt builder API routes."""

from fastapi import APIRouter, Query

from src.api.dependencies import PromptBuilderServiceDep
from src.api.schemas.prompt_builder import (
    PromptSectionCreate,
    PromptSectionResponse,
    PromptSectionUpdate,
    PromptSectionValueCreate,
    PromptSectionValueResponse,
    PromptSectionValueUpdate,
    PromptTemplateCreate,
    PromptTemplateDetailResponse,
    PromptTemplateResponse,
    PromptTemplateSave,
    PromptTemplateSectionCreate,
    PromptTemplateSectionReorder,
    PromptTemplateSectionResponse,
    PromptTemplateSectionUpdate,
    PromptTemplateUpdate,
)
from src.db.tables.prompt_builder import PromptPolarity
from src.services.prompt_builder.compose import resolve_selected_value_ids
from src.services.prompt_builder.service import (
    PromptTemplateDetail,
)

router = APIRouter(tags=["prompt-builder"])


def _section_response(detail) -> PromptSectionResponse:
    return PromptSectionResponse.model_validate(
        {
            **detail.section.model_dump(),
            "values": detail.values,
        }
    )


def _template_detail_response(detail: PromptTemplateDetail) -> PromptTemplateDetailResponse:
    return PromptTemplateDetailResponse(
        id=detail.template.id,  # type: ignore[arg-type]
        name=detail.template.name,
        polarity=detail.template.polarity,
        created_at=detail.template.created_at,
        updated_at=detail.template.updated_at,
        composed_prompt=detail.composed_prompt,
        sections=[
            PromptTemplateSectionResponse(
                id=entry.link.id,  # type: ignore[arg-type]
                template_id=entry.link.template_id,
                section_id=entry.link.section_id,
                section_name=entry.section.name,
                sort_order=entry.link.sort_order,
                selection_mode=entry.link.selection_mode,
                selected_value_id=entry.link.selected_value_id,
                selected_value_ids=resolve_selected_value_ids(entry.link),
                include_empty=entry.link.include_empty,
                values=[
                    PromptSectionValueResponse.model_validate(value) for value in entry.values
                ],
            )
            for entry in detail.sections
        ],
    )


@router.get("/prompt-sections/", response_model=list[PromptSectionResponse])
async def list_prompt_sections(service: PromptBuilderServiceDep) -> list[PromptSectionResponse]:
    return [_section_response(detail) for detail in await service.list_sections()]


@router.post("/prompt-sections/", response_model=PromptSectionResponse, status_code=201)
async def create_prompt_section(
    body: PromptSectionCreate,
    service: PromptBuilderServiceDep,
) -> PromptSectionResponse:
    detail = await service.create_section(name=body.name)
    return _section_response(detail)


@router.patch("/prompt-sections/{section_id}", response_model=PromptSectionResponse)
async def update_prompt_section(
    section_id: int,
    body: PromptSectionUpdate,
    service: PromptBuilderServiceDep,
) -> PromptSectionResponse:
    section = await service.update_section(section_id, name=body.name)
    values = await service.list_section_values(section_id)
    return PromptSectionResponse.model_validate({**section.model_dump(), "values": values})


@router.delete("/prompt-sections/{section_id}", status_code=204)
async def delete_prompt_section(section_id: int, service: PromptBuilderServiceDep) -> None:
    await service.delete_section(section_id)


@router.get("/prompt-sections/{section_id}/values/", response_model=list[PromptSectionValueResponse])
async def list_prompt_section_values(
    section_id: int,
    service: PromptBuilderServiceDep,
) -> list[PromptSectionValueResponse]:
    values = await service.list_section_values(section_id)
    return [PromptSectionValueResponse.model_validate(value) for value in values]


@router.post(
    "/prompt-sections/{section_id}/values/",
    response_model=PromptSectionValueResponse,
    status_code=201,
)
async def create_prompt_section_value(
    section_id: int,
    body: PromptSectionValueCreate,
    service: PromptBuilderServiceDep,
) -> PromptSectionValueResponse:
    value = await service.create_section_value(section_id, text=body.text)
    return PromptSectionValueResponse.model_validate(value)


@router.patch(
    "/prompt-sections/{section_id}/values/{value_id}",
    response_model=PromptSectionValueResponse,
)
async def update_prompt_section_value(
    section_id: int,
    value_id: int,
    body: PromptSectionValueUpdate,
    service: PromptBuilderServiceDep,
) -> PromptSectionValueResponse:
    _ = section_id
    value = await service.update_section_value(value_id, text=body.text)
    return PromptSectionValueResponse.model_validate(value)


@router.delete("/prompt-sections/{section_id}/values/{value_id}", status_code=204)
async def delete_prompt_section_value(
    section_id: int,
    value_id: int,
    service: PromptBuilderServiceDep,
) -> None:
    _ = section_id
    await service.delete_section_value(value_id)


@router.get("/prompt-workspace/{polarity}", response_model=PromptTemplateDetailResponse)
async def get_prompt_workspace(
    polarity: PromptPolarity,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.get_workspace(polarity)
    return _template_detail_response(detail)


@router.post(
    "/prompt-workspace/{polarity}/sections/",
    response_model=PromptTemplateDetailResponse,
    status_code=201,
)
async def add_prompt_workspace_section(
    polarity: PromptPolarity,
    body: PromptTemplateSectionCreate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.add_workspace_section(polarity, section_id=body.section_id)
    return _template_detail_response(detail)


@router.patch(
    "/prompt-workspace/{polarity}/sections/reorder",
    response_model=PromptTemplateDetailResponse,
)
async def reorder_prompt_workspace_sections(
    polarity: PromptPolarity,
    body: PromptTemplateSectionReorder,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.reorder_workspace_sections(polarity, order=body.order)
    return _template_detail_response(detail)


@router.patch(
    "/prompt-workspace/{polarity}/sections/{link_id}",
    response_model=PromptTemplateDetailResponse,
)
async def update_prompt_workspace_section(
    polarity: PromptPolarity,
    link_id: int,
    body: PromptTemplateSectionUpdate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.update_workspace_section(
        polarity,
        link_id,
        selection_mode=body.selection_mode,
        selected_value_id=body.selected_value_id,
        selected_value_ids=body.selected_value_ids,
        clear_selected_value=body.clear_selected_value,
        include_empty=body.include_empty,
    )
    return _template_detail_response(detail)


@router.delete("/prompt-workspace/{polarity}/sections/{link_id}", status_code=204)
async def delete_prompt_workspace_section(
    polarity: PromptPolarity,
    link_id: int,
    service: PromptBuilderServiceDep,
) -> None:
    await service.remove_workspace_section(polarity, link_id)


@router.post(
    "/prompt-workspace/{polarity}/save",
    response_model=PromptTemplateResponse,
    status_code=201,
)
async def save_prompt_workspace_as_template(
    polarity: PromptPolarity,
    body: PromptTemplateSave,
    service: PromptBuilderServiceDep,
) -> PromptTemplateResponse:
    template = await service.save_workspace_as_template(polarity=polarity, name=body.name)
    return PromptTemplateResponse.model_validate(template)


@router.post("/prompt-templates/{template_id}/load", response_model=PromptTemplateDetailResponse)
async def load_prompt_template_into_workspace(
    template_id: int,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.load_template_into_workspace(template_id)
    return _template_detail_response(detail)


@router.get("/prompt-templates/", response_model=list[PromptTemplateResponse])
async def list_prompt_templates(
    service: PromptBuilderServiceDep,
    polarity: PromptPolarity | None = Query(default=None),
) -> list[PromptTemplateResponse]:
    templates = await service.list_templates(polarity)
    return [PromptTemplateResponse.model_validate(template) for template in templates]


@router.post("/prompt-templates/", response_model=PromptTemplateResponse, status_code=201)
async def create_prompt_template(
    body: PromptTemplateCreate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateResponse:
    template = await service.create_template(name=body.name, polarity=body.polarity)
    return PromptTemplateResponse.model_validate(template)


@router.get("/prompt-templates/{template_id}", response_model=PromptTemplateDetailResponse)
async def get_prompt_template(
    template_id: int,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.get_template_detail(template_id)
    return _template_detail_response(detail)


@router.patch("/prompt-templates/{template_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(
    template_id: int,
    body: PromptTemplateUpdate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateResponse:
    template = await service.update_template(template_id, name=body.name)
    return PromptTemplateResponse.model_validate(template)


@router.delete("/prompt-templates/{template_id}", status_code=204)
async def delete_prompt_template(template_id: int, service: PromptBuilderServiceDep) -> None:
    await service.delete_template(template_id)


@router.post(
    "/prompt-templates/{template_id}/sections/",
    response_model=PromptTemplateDetailResponse,
    status_code=201,
)
async def add_prompt_template_section(
    template_id: int,
    body: PromptTemplateSectionCreate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.add_template_section(template_id, section_id=body.section_id)
    return _template_detail_response(detail)


@router.patch(
    "/prompt-templates/{template_id}/sections/reorder",
    response_model=PromptTemplateDetailResponse,
)
async def reorder_prompt_template_sections(
    template_id: int,
    body: PromptTemplateSectionReorder,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.reorder_template_sections(template_id, order=body.order)
    return _template_detail_response(detail)


@router.patch(
    "/prompt-templates/{template_id}/sections/{link_id}",
    response_model=PromptTemplateDetailResponse,
)
async def update_prompt_template_section(
    template_id: int,
    link_id: int,
    body: PromptTemplateSectionUpdate,
    service: PromptBuilderServiceDep,
) -> PromptTemplateDetailResponse:
    detail = await service.update_template_section(
        template_id,
        link_id,
        selection_mode=body.selection_mode,
        selected_value_id=body.selected_value_id,
        selected_value_ids=body.selected_value_ids,
        clear_selected_value=body.clear_selected_value,
        include_empty=body.include_empty,
    )
    return _template_detail_response(detail)


@router.delete("/prompt-templates/{template_id}/sections/{link_id}", status_code=204)
async def delete_prompt_template_section(
    template_id: int,
    link_id: int,
    service: PromptBuilderServiceDep,
) -> None:
    await service.remove_template_section(template_id, link_id)
