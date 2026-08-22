"""Shared runnable sample URL/response helpers."""

from pathlib import Path

from fastapi import FastAPI
from src.api.routers.loras import router as lora_router
from src.api.routers.runnable import build_sample_responses, sample_file_url
from src.api.routers.samplings import router as sampling_router

_SHARED_SUFFIXES = (
    "/",
    "/{entity_id}",
    "/{entity_id}/enqueue",
    "/{entity_id}/cancel",
    "/{entity_id}/logs",
    "/{entity_id}/samples",
    "/{entity_id}/sample-file/{file_path:path}",
)


def _route_paths(router) -> set[str]:
    return {getattr(route, "path", "") for route in router.routes}


def test_sampling_live_preview_route_exists() -> None:
    assert "/samplings/{entity_id}/live-preview" in _route_paths(sampling_router)


def test_lora_and_sampling_share_runnable_paths() -> None:
    lora_paths = _route_paths(lora_router)
    sampling_paths = _route_paths(sampling_router)
    for suffix in _SHARED_SUFFIXES:
        assert f"/loras{suffix}" in lora_paths
        assert f"/samplings{suffix}" in sampling_paths


def test_runnable_operation_ids_are_unique() -> None:
    app = FastAPI()
    app.include_router(lora_router)
    app.include_router(sampling_router)
    operation_ids = [
        operation["operationId"]
        for path_item in app.openapi()["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert operation_ids
    assert len(operation_ids) == len(set(operation_ids))


def test_sample_file_url_normalizes_separators() -> None:
    assert sample_file_url("loras", 3, r"samples\a.png") == "/loras/3/sample-file/samples/a.png"
    assert sample_file_url("/samplings", 9, "grid.png") == "/samplings/9/sample-file/grid.png"


def test_build_sample_responses_empty_without_output_dir(tmp_path: Path) -> None:
    sample = tmp_path / "a.png"
    sample.write_bytes(b"x")
    result = build_sample_responses(1, "loras", None, [(sample, "legacy", {})])
    assert result.samples == []


def test_build_sample_responses_builds_urls(tmp_path: Path) -> None:
    nested = tmp_path / "samples"
    nested.mkdir()
    sample = nested / "a.png"
    sample.write_bytes(b"x")

    result = build_sample_responses(4, "loras", tmp_path, [(sample, "legacy", {"i": 1})])

    assert len(result.samples) == 1
    item = result.samples[0]
    assert item.filename == "a.png"
    assert item.path == str(sample)
    assert item.url == "/loras/4/sample-file/samples/a.png"
    assert item.kind == "legacy"
    assert item.metadata == {"i": 1}
