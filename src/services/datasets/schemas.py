"""Service-layer DTOs for dataset operations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class DatasetUpdateRequest:
    name: str | None = None
    relative_path: str | None = None
    description: str | None = None
    target_resolution: int | None = None
    enable_bucket: bool | None = None
    bucket_reso_steps: int | None = None
    min_bucket_reso: int | None = None
    max_bucket_reso: int | None = None
    bucket_no_upscale: bool | None = None
    provided: frozenset[str] = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_fields(cls, **raw_fields: object) -> DatasetUpdateRequest:
        valid = {item.name for item in fields(cls) if item.name != "provided"}
        provided = frozenset(key for key in raw_fields if key in valid)
        return cls(
            provided=provided,
            **{key: value for key, value in raw_fields.items() if key in valid},
        )

    def was_provided(self, field_name: str) -> bool:
        return field_name in self.provided
