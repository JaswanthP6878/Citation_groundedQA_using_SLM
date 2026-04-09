from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.config import get_default_project_paths


def main() -> None:
    paths = get_default_project_paths(repo_root=REPO_ROOT)
    print(json.dumps(paths.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
