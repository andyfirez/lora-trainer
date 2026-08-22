"""Exceptions for the prompt builder service."""

from src.services.common.exceptions import AppError, NotFoundError


class PromptSectionNotFoundError(NotFoundError):
    def __init__(self, section_id: int) -> None:
        super().__init__("Prompt section", section_id)


class PromptSectionValueNotFoundError(NotFoundError):
    def __init__(self, value_id: int) -> None:
        super().__init__("Prompt section value", value_id)


class PromptTemplateNotFoundError(NotFoundError):
    def __init__(self, template_id: int) -> None:
        super().__init__("Prompt template", template_id)


class PromptTemplateSectionNotFoundError(NotFoundError):
    def __init__(self, link_id: int) -> None:
        super().__init__("Prompt template section", link_id)


class PromptSectionInUseError(AppError):
    status_code = 409

    def __init__(self, section_id: int) -> None:
        super().__init__(f"Prompt section {section_id} is used by one or more templates")
        self.section_id = section_id


class PromptSectionValueInUseError(AppError):
    status_code = 409

    def __init__(self, value_id: int) -> None:
        super().__init__(f"Prompt section value {value_id} is selected in a template")
        self.value_id = value_id


class PromptSectionAlreadyLinkedError(AppError):
    status_code = 409

    def __init__(self, section_id: int, template_id: int) -> None:
        super().__init__(f"Section {section_id} is already linked to template {template_id}")
        self.section_id = section_id
        self.template_id = template_id


class PromptWorkspaceTemplateError(AppError):
    status_code = 409

    def __init__(self, message: str = "Workspace templates cannot be modified this way") -> None:
        super().__init__(message)


class PromptTemplatePolarityMismatchError(AppError):
    status_code = 400

    def __init__(self, template_id: int, expected: str, actual: str) -> None:
        super().__init__(
            f"Template {template_id} has polarity '{actual}', expected '{expected}'"
        )
        self.template_id = template_id


class InvalidPromptValueTextError(AppError):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
