from pathlib import Path
import os
from unittest.mock import patch

from qasper_rag.config import (
    DEFAULT_CHUNKING_SPECS,
    _resolve_scc_shared_project_root,
    get_default_project_paths,
)


def test_default_project_paths_fall_back_to_repo_local_data() -> None:
    fake_repo_root = Path("/tmp/fake-repo")
    paths = get_default_project_paths(repo_root=fake_repo_root)

    assert paths.repo_root == fake_repo_root.resolve()
    assert "qasper" in str(paths.shared_qasper_root)
    assert paths.qasper_processed_dir.name == "processed"
    assert DEFAULT_CHUNKING_SPECS["fixed"].max_tokens == 256
    assert DEFAULT_CHUNKING_SPECS["sliding"].overlap_tokens == 128
    assert DEFAULT_CHUNKING_SPECS["section"].max_tokens == 512


def test_default_project_paths_respect_env_overrides() -> None:
    fake_repo_root = Path("/tmp/fake-repo")
    old_user_root = os.environ.get("NLP_PROJECT_USER_ROOT")
    old_shared_root = os.environ.get("NLP_PROJECT_SHARED_ROOT")
    try:
        os.environ["NLP_PROJECT_USER_ROOT"] = "/tmp/custom-user-root"
        os.environ["NLP_PROJECT_SHARED_ROOT"] = "/tmp/custom-shared-root"
        paths = get_default_project_paths(repo_root=fake_repo_root)
    finally:
        if old_user_root is None:
            os.environ.pop("NLP_PROJECT_USER_ROOT", None)
        else:
            os.environ["NLP_PROJECT_USER_ROOT"] = old_user_root
        if old_shared_root is None:
            os.environ.pop("NLP_PROJECT_SHARED_ROOT", None)
        else:
            os.environ["NLP_PROJECT_SHARED_ROOT"] = old_shared_root

    assert str(paths.user_project_root) == "/tmp/custom-user-root"
    assert str(paths.shared_project_root) == "/tmp/custom-shared-root"


def test_resolve_scc_shared_project_root_prefers_renamed_project() -> None:
    target = Path("/projectnb/cs505am/projects/zukhriddin_nlp_research_project")

    def fake_exists(path: Path) -> bool:
        return path == target

    with patch("pathlib.Path.exists", new=fake_exists):
        resolved = _resolve_scc_shared_project_root()

    assert resolved == target
