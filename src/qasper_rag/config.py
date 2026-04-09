from __future__ import annotations

from dataclasses import asdict, dataclass
import getpass
import os
from pathlib import Path

from .chunking import FixedChunker, SectionAwareChunker, SlidingWindowChunker

DEFAULT_SHARED_PROJECT_NAMES = (
    "zukhriddin_nlp_research_project",
    "nlp_research_project",
)
DEFAULT_USER_PROJECT_NAMES = (
    "zukhriddin_nlp_research_project",
    "nlp_research_project",
)


@dataclass(frozen=True)
class ChunkingSpec:
    name: str
    max_tokens: int
    overlap_tokens: int
    include_abstract: bool = True
    include_figures_and_tables: bool = False

    def build_chunker(self):
        if self.name == "fixed":
            return FixedChunker(
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                include_abstract=self.include_abstract,
                include_figures_and_tables=self.include_figures_and_tables,
            )
        if self.name == "sliding":
            return SlidingWindowChunker(
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                include_abstract=self.include_abstract,
                include_figures_and_tables=self.include_figures_and_tables,
            )
        if self.name == "section":
            return SectionAwareChunker(
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                include_abstract=self.include_abstract,
                include_figures_and_tables=self.include_figures_and_tables,
            )
        raise ValueError(f"Unsupported chunking strategy: {self.name}")


DEFAULT_CHUNKING_SPECS = {
    "fixed": ChunkingSpec(name="fixed", max_tokens=256, overlap_tokens=32),
    "sliding": ChunkingSpec(name="sliding", max_tokens=512, overlap_tokens=128),
    "section": ChunkingSpec(name="section", max_tokens=512, overlap_tokens=0),
}


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    shared_project_root: Path
    shared_qasper_root: Path
    qasper_raw_dir: Path
    qasper_extracted_dir: Path
    qasper_processed_dir: Path
    user_project_root: Path
    user_runs_dir: Path
    user_indexes_dir: Path
    hf_home: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def get_default_project_paths(repo_root: str | Path | None = None) -> ProjectPaths:
    resolved_repo_root = Path(repo_root).resolve() if repo_root is not None else _detect_repo_root()
    username = os.environ.get("USER") or getpass.getuser() or "unknown"

    if Path("/projectnb/cs505am").exists():
        shared_project_root = _resolve_scc_shared_project_root()
        user_project_root = _resolve_scc_user_project_root(username)
        hf_home = Path(
            os.environ.get(
                "HF_HOME",
                f"/projectnb/cs505am/students/{username}/.cache/huggingface",
            )
        )
    else:
        shared_project_root = Path(
            os.environ.get(
                "NLP_PROJECT_SHARED_ROOT",
                str(resolved_repo_root / "data" / "shared" / "nlp_research_project"),
            )
        )
        user_project_root = Path(
            os.environ.get(
                "NLP_PROJECT_USER_ROOT",
                str(resolved_repo_root / "data" / "users" / username / "nlp_research_project"),
            )
        )
        hf_home = Path(
            os.environ.get(
                "HF_HOME",
                str(resolved_repo_root / "data" / "users" / username / ".cache" / "huggingface"),
            )
        )

    shared_qasper_root = Path(
        os.environ.get("NLP_PROJECT_QASPER_ROOT", str(shared_project_root / "qasper"))
    )

    return ProjectPaths(
        repo_root=resolved_repo_root,
        shared_project_root=shared_project_root,
        shared_qasper_root=shared_qasper_root,
        qasper_raw_dir=shared_qasper_root / "raw",
        qasper_extracted_dir=shared_qasper_root / "extracted",
        qasper_processed_dir=shared_qasper_root / "processed",
        user_project_root=user_project_root,
        user_runs_dir=user_project_root / "runs",
        user_indexes_dir=user_project_root / "indexes",
        hf_home=hf_home,
    )


def ensure_project_layout(paths: ProjectPaths, *, create_shared: bool = False) -> None:
    user_directories = [
        paths.user_project_root,
        paths.user_runs_dir,
        paths.user_indexes_dir,
        paths.hf_home,
    ]
    for directory in user_directories:
        directory.mkdir(parents=True, exist_ok=True)

    if create_shared:
        shared_directories = [
            paths.shared_project_root,
            paths.shared_qasper_root,
            paths.qasper_raw_dir,
            paths.qasper_extracted_dir,
            paths.qasper_processed_dir,
        ]
        for directory in shared_directories:
            directory.mkdir(parents=True, exist_ok=True)


def _detect_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_scc_user_project_root(username: str) -> Path:
    env_override = os.environ.get("NLP_PROJECT_USER_ROOT")
    if env_override:
        return Path(env_override)

    student_root = Path(f"/projectnb/cs505am/students/{username}")
    for project_name in DEFAULT_USER_PROJECT_NAMES:
        candidate = student_root / project_name
        if candidate.exists():
            return candidate

    return student_root / DEFAULT_USER_PROJECT_NAMES[0]


def _resolve_scc_shared_project_root() -> Path:
    env_override = os.environ.get("NLP_PROJECT_SHARED_ROOT")
    if env_override:
        return Path(env_override)

    shared_root = Path("/projectnb/cs505am/projects")
    for project_name in DEFAULT_SHARED_PROJECT_NAMES:
        candidate = shared_root / project_name
        if candidate.exists():
            return candidate

    return shared_root / DEFAULT_SHARED_PROJECT_NAMES[0]
