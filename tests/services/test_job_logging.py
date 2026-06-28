from src.services.jobs.job_logging import build_job_log_path


def test_build_job_log_path() -> None:
    path = build_job_log_path(42)
    assert path.name == "job_42.log"
    assert path.parent.name == "logs"
