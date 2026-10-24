"""Create and deploy OOCSS to a Hugging Face Docker Space.

Authenticate first with `hf auth login`, then run:
    python deploy_to_hf.py OWNER/SPACE-NAME

The Space is created as public, matching the project's requested public release.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
SPACE_CARD = """---
title: OOCSS Oral Cancer Screening System
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

"""
EXCLUDES = [
    ".git/**",
    ".venv/**",
    "venv/**",
    "__pycache__/**",
    "*.py[cod]",
    ".env",
    ".env.*",
    ".kaggle/**",
    ".vscode/**",
    ".idea/**",
    "data/**",
    "outputs/**",
    "static/*.docx",
    "README.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space_id", help="Hugging Face Space ID, for example owner/oocss-demo")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Allow uploading to an existing Space with this exact ID.",
    )
    args = parser.parse_args()

    api = HfApi()
    try:
        api.repo_info(repo_id=args.space_id, repo_type="space")
        exists = True
    except Exception as exc:
        if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
            exists = False
        else:
            raise

    if exists and not args.update_existing:
        raise SystemExit(
            f"Space {args.space_id} already exists. Review it first, then rerun with --update-existing."
        )

    if not exists:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
        )

    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=ROOT,
        ignore_patterns=EXCLUDES,
        commit_message="Deploy OOCSS Docker Space",
    )

    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO((SPACE_CARD + readme).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.space_id,
        repo_type="space",
        commit_message="Add Hugging Face Space card metadata",
    )
    print(f"Deployment uploaded: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
