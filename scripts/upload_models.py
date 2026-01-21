#!/usr/bin/env python3
"""
Upload Models to GitHub Releases.

Packages trained models into a tarball and uploads to GitHub Releases
for deployment to Streamlit Cloud.

Usage:
    python scripts/upload_models.py                    # Upload with auto-generated tag
    python scripts/upload_models.py --tag v1.0.0      # Upload with specific tag
    python scripts/upload_models.py --s3              # Upload to S3 instead

Environment Variables:
    GITHUB_TOKEN: Personal access token with repo scope
    GITHUB_REPO: Repository in owner/repo format (default: from git remote)
    S3_BUCKET: S3 bucket name (for --s3 mode)
    AWS_ACCESS_KEY_ID: AWS credentials (for --s3 mode)
    AWS_SECRET_ACCESS_KEY: AWS credentials (for --s3 mode)
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Constants
MODELS_ARCHIVE_NAME = "models.tar.gz"
DATA_ARCHIVE_NAME = "data.tar.gz"


def get_github_repo() -> str:
    """Get GitHub repository from environment or git remote."""
    repo = os.environ.get("GITHUB_REPO")
    if repo:
        return repo

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Parse URL: https://github.com/owner/repo.git or git@github.com:owner/repo.git
            if "github.com" in url:
                if url.startswith("git@"):
                    # git@github.com:owner/repo.git
                    repo = url.split(":")[-1].replace(".git", "")
                else:
                    # https://github.com/owner/repo.git
                    parts = url.split("github.com/")[-1].replace(".git", "")
                    repo = parts
                return repo
    except Exception:
        pass

    raise ValueError(
        "Could not determine GitHub repository. "
        "Set GITHUB_REPO environment variable or run from a git repository."
    )


def get_github_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN environment variable not set. "
            "Create a token at: https://github.com/settings/tokens"
        )
    return token


def generate_tag() -> str:
    """Generate a version tag based on current timestamp."""
    now = datetime.now()
    return f"models-{now.strftime('%Y%m%d-%H%M%S')}"


def package_models(output_path: Path) -> bool:
    """
    Package models directory into a tarball.

    Args:
        output_path: Path for the output tar.gz file.

    Returns:
        True if successful, False otherwise.
    """
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found: {MODELS_DIR}")
        return False

    # Count models
    symbols = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir():
            classifier = path / "model_classifier.joblib"
            regressor = path / "model_regressor.joblib"
            if classifier.exists() and regressor.exists():
                symbols.append(path.name)

    if not symbols:
        print("Error: No trained models found")
        return False

    print(f"Packaging models for {len(symbols)} symbols: {', '.join(symbols)}")

    # Create tarball
    with tarfile.open(output_path, "w:gz") as tar:
        for symbol in symbols:
            symbol_dir = MODELS_DIR / symbol
            for file in symbol_dir.iterdir():
                # Add file with relative path inside archive
                arcname = f"{symbol}/{file.name}"
                tar.add(file, arcname=arcname)
                print(f"  Added: {arcname}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Created archive: {output_path.name} ({size_mb:.2f} MB)")

    return True


def package_data(output_path: Path) -> bool:
    """
    Package CSV data files into a tarball.

    Args:
        output_path: Path for the output tar.gz file.

    Returns:
        True if successful, False otherwise.
    """
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"

    if not raw_dir.exists() and not processed_dir.exists():
        print("Warning: No data directories found")
        return False

    with tarfile.open(output_path, "w:gz") as tar:
        # Add raw CSV files
        if raw_dir.exists():
            for file in raw_dir.glob("*.csv"):
                arcname = f"raw/{file.name}"
                tar.add(file, arcname=arcname)
                print(f"  Added: {arcname}")

        # Add processed CSV files
        if processed_dir.exists():
            for file in processed_dir.glob("*.csv"):
                arcname = f"processed/{file.name}"
                tar.add(file, arcname=arcname)
                print(f"  Added: {arcname}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Created archive: {output_path.name} ({size_mb:.2f} MB)")

    return True


def create_github_release(repo: str, token: str, tag: str) -> Optional[dict]:
    """
    Create a GitHub release.

    Args:
        repo: Repository in owner/repo format.
        token: GitHub personal access token.
        tag: Release tag.

    Returns:
        Release data dict if successful, None otherwise.
    """
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check if release already exists
    check_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    response = requests.get(check_url, headers=headers)
    if response.status_code == 200:
        print(f"Release {tag} already exists, updating...")
        return response.json()

    # Create new release
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    data = {
        "tag_name": tag,
        "name": f"Models {tag}",
        "body": f"ML models uploaded at {now}\n\nDownload models.tar.gz and extract to data/models/",
        "draft": False,
        "prerelease": False,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        print(f"Created release: {tag}")
        return response.json()
    else:
        print(f"Error creating release: {response.status_code}")
        print(response.json())
        return None


def upload_asset(
    repo: str,
    token: str,
    release_id: int,
    asset_path: Path,
    asset_name: str
) -> bool:
    """
    Upload an asset to a GitHub release.

    Args:
        repo: Repository in owner/repo format.
        token: GitHub personal access token.
        release_id: ID of the release.
        asset_path: Path to the file to upload.
        asset_name: Name for the asset in the release.

    Returns:
        True if successful, False otherwise.
    """
    # Check if asset already exists and delete it
    list_url = f"https://api.github.com/repos/{repo}/releases/{release_id}/assets"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(list_url, headers=headers)
    if response.status_code == 200:
        for asset in response.json():
            if asset["name"] == asset_name:
                delete_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
                requests.delete(delete_url, headers=headers)
                print(f"Deleted existing asset: {asset_name}")

    # Upload new asset
    upload_url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/gzip",
    }
    params = {"name": asset_name}

    with open(asset_path, "rb") as f:
        response = requests.post(
            upload_url,
            headers=headers,
            params=params,
            data=f,
        )

    if response.status_code == 201:
        print(f"Uploaded: {asset_name}")
        return True
    else:
        print(f"Error uploading {asset_name}: {response.status_code}")
        print(response.json())
        return False


def upload_to_github(tag: str, include_data: bool = False) -> bool:
    """
    Upload models (and optionally data) to GitHub Releases.

    Args:
        tag: Release tag.
        include_data: If True, also upload CSV data files.

    Returns:
        True if successful, False otherwise.
    """
    repo = get_github_repo()
    token = get_github_token()

    print(f"Repository: {repo}")
    print(f"Tag: {tag}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Package models
        models_archive = tmpdir / MODELS_ARCHIVE_NAME
        if not package_models(models_archive):
            return False

        # Package data if requested
        data_archive = None
        if include_data:
            data_archive = tmpdir / DATA_ARCHIVE_NAME
            package_data(data_archive)

        # Create release
        release = create_github_release(repo, token, tag)
        if not release:
            return False

        release_id = release["id"]

        # Upload models
        if not upload_asset(repo, token, release_id, models_archive, MODELS_ARCHIVE_NAME):
            return False

        # Upload data if packaged
        if data_archive and data_archive.exists():
            upload_asset(repo, token, release_id, data_archive, DATA_ARCHIVE_NAME)

    print(f"\nSuccess! Models uploaded to:")
    print(f"  https://github.com/{repo}/releases/tag/{tag}")
    print(f"\nTo update the dashboard, it will automatically fetch new models.")

    return True


def upload_to_s3(include_data: bool = False) -> bool:
    """
    Upload models (and optionally data) to S3.

    Args:
        include_data: If True, also upload CSV data files.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import boto3
    except ImportError:
        print("Error: boto3 not installed. Run: pip install boto3")
        return False

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("Error: S3_BUCKET environment variable not set")
        return False

    print(f"S3 Bucket: {bucket}")

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Package and upload models
        models_archive = tmpdir / MODELS_ARCHIVE_NAME
        if not package_models(models_archive):
            return False

        print(f"Uploading to s3://{bucket}/models/{MODELS_ARCHIVE_NAME}")
        s3.upload_file(str(models_archive), bucket, f"models/{MODELS_ARCHIVE_NAME}")

        # Upload version marker
        version = datetime.now().isoformat()
        version_file = tmpdir / "version.txt"
        version_file.write_text(version)
        s3.upload_file(str(version_file), bucket, "models/version.txt")

        # Package and upload data if requested
        if include_data:
            data_archive = tmpdir / DATA_ARCHIVE_NAME
            if package_data(data_archive):
                print(f"Uploading to s3://{bucket}/data/{DATA_ARCHIVE_NAME}")
                s3.upload_file(str(data_archive), bucket, f"data/{DATA_ARCHIVE_NAME}")

    print(f"\nSuccess! Models uploaded to s3://{bucket}/models/")

    return True


def list_models() -> None:
    """List currently trained models."""
    if not MODELS_DIR.exists():
        print("No models directory found")
        return

    print("Trained Models:")
    print("-" * 60)

    for path in sorted(MODELS_DIR.iterdir()):
        if not path.is_dir():
            continue

        metadata_path = path / "metadata.json"
        classifier_path = path / "model_classifier.joblib"
        regressor_path = path / "model_regressor.joblib"

        symbol = path.name

        if not classifier_path.exists() or not regressor_path.exists():
            print(f"  {symbol}: INCOMPLETE (missing model files)")
            continue

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            trained_at = metadata.get("trained_at", "unknown")
            n_samples = metadata.get("n_samples", "?")
            accuracy = metadata.get("classifier_metrics", {}).get("accuracy_mean")
            accuracy_str = f"{accuracy:.1%}" if accuracy else "N/A"
            print(f"  {symbol}: trained={trained_at[:16]}, samples={n_samples}, accuracy={accuracy_str}")
        else:
            print(f"  {symbol}: OK (no metadata)")


def main():
    parser = argparse.ArgumentParser(
        description="Upload ML models to GitHub Releases or S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_models.py              # Upload to GitHub with auto tag
  python scripts/upload_models.py --tag v1.0   # Upload with specific tag
  python scripts/upload_models.py --s3         # Upload to S3
  python scripts/upload_models.py --list       # List trained models
  python scripts/upload_models.py --with-data  # Include CSV data files
        """,
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Upload to S3 instead of GitHub",
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Also upload CSV data files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List trained models without uploading",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Generate tag if not provided
    tag = args.tag or generate_tag()

    # Upload
    if args.s3:
        success = upload_to_s3(include_data=args.with_data)
    else:
        success = upload_to_github(tag, include_data=args.with_data)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
