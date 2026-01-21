"""
Remote Model Loader.

Downloads ML models from GitHub Releases for production deployment.
Supports caching and version checking to minimize unnecessary downloads.
"""

import os
import json
import tarfile
import tempfile
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

import requests


logger = logging.getLogger(__name__)

# Environment variables
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
MODEL_STORAGE_ENV = "MODEL_STORAGE"
S3_BUCKET_ENV = "S3_BUCKET"

# Default settings
DEFAULT_REPO = "Hussain0327/Market-Sentiment-Risk-Analytics"
DEFAULT_ASSET_NAME = "models.tar.gz"
CACHE_CHECK_INTERVAL = timedelta(hours=1)  # How often to check for new models


class RemoteModelLoader:
    """
    Loads ML models from remote storage (GitHub Releases or S3).

    Features:
    - Downloads models from GitHub Releases
    - Caches models locally in temp directory
    - Checks for updates periodically
    - Falls back to local models if available

    Example:
        >>> loader = RemoteModelLoader()
        >>> models_dir = loader.get_models_dir()
        >>> # Use models from models_dir
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        github_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        storage_type: Optional[str] = None,
    ):
        """
        Initialize the remote model loader.

        Args:
            repo: GitHub repository (owner/repo format).
            github_token: GitHub personal access token for private repos.
            cache_dir: Directory for caching downloaded models.
            storage_type: Storage backend ("github" or "s3").
        """
        self.repo = repo or os.environ.get("GITHUB_REPO", DEFAULT_REPO)
        self.github_token = github_token or os.environ.get(GITHUB_TOKEN_ENV)
        self.storage_type = storage_type or os.environ.get(MODEL_STORAGE_ENV, "github")

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use system temp with app-specific subdirectory
            self.cache_dir = Path(tempfile.gettempdir()) / "market_sentiment_models"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir = self.cache_dir / "models"
        self._metadata_file = self.cache_dir / "remote_metadata.json"
        self._last_check_file = self.cache_dir / "last_check.txt"

        # Local fallback
        self._local_models_dir = Path(__file__).parent.parent / "data" / "models"

    def get_models_dir(self, force_refresh: bool = False) -> Path:
        """
        Get the path to models directory, downloading if needed.

        Args:
            force_refresh: If True, always check for and download updates.

        Returns:
            Path to directory containing model files.
        """
        # Check if we need to download/update
        if force_refresh or self._should_check_for_updates():
            try:
                self._update_models()
            except Exception as e:
                logger.warning(f"Failed to update models from remote: {e}")

        # Return cached models if available
        if self._models_dir.exists() and self._has_valid_models():
            return self._models_dir

        # Fall back to local models
        if self._local_models_dir.exists():
            logger.info("Using local models as fallback")
            return self._local_models_dir

        # No models available
        raise RuntimeError(
            "No models available. Either download from remote or train locally."
        )

    def get_trained_symbols(self) -> List[str]:
        """Get list of symbols with available models."""
        models_dir = self.get_models_dir()
        symbols = []

        for path in models_dir.iterdir():
            if path.is_dir():
                metadata = path / "metadata.json"
                if metadata.exists():
                    symbols.append(path.name)

        return sorted(symbols)

    def load_model_metadata(self, symbol: str) -> Optional[Dict]:
        """Load metadata for a specific symbol's model."""
        models_dir = self.get_models_dir()
        metadata_path = models_dir / symbol / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def get_model_paths(self, symbol: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Get paths to classifier and regressor model files.

        Returns:
            Tuple of (classifier_path, regressor_path), either may be None.
        """
        models_dir = self.get_models_dir()
        model_dir = models_dir / symbol

        classifier_path = model_dir / "model_classifier.joblib"
        regressor_path = model_dir / "model_regressor.joblib"

        return (
            classifier_path if classifier_path.exists() else None,
            regressor_path if regressor_path.exists() else None
        )

    def get_remote_version(self) -> Optional[str]:
        """Get version/tag of latest remote models."""
        if self.storage_type == "github":
            return self._get_github_latest_release()
        elif self.storage_type == "s3":
            return self._get_s3_latest_version()
        return None

    def get_cached_version(self) -> Optional[str]:
        """Get version of currently cached models."""
        if not self._metadata_file.exists():
            return None

        try:
            with open(self._metadata_file) as f:
                metadata = json.load(f)
            return metadata.get("version")
        except Exception:
            return None

    def _should_check_for_updates(self) -> bool:
        """Determine if we should check remote for updates."""
        # Always check if no cached models
        if not self._models_dir.exists() or not self._has_valid_models():
            return True

        # Check if enough time has passed since last check
        if not self._last_check_file.exists():
            return True

        try:
            with open(self._last_check_file) as f:
                last_check = datetime.fromisoformat(f.read().strip())
            return datetime.now() - last_check > CACHE_CHECK_INTERVAL
        except Exception:
            return True

    def _has_valid_models(self) -> bool:
        """Check if cached models directory has valid models."""
        if not self._models_dir.exists():
            return False

        # Check for at least one symbol with models
        for path in self._models_dir.iterdir():
            if path.is_dir():
                classifier = path / "model_classifier.joblib"
                regressor = path / "model_regressor.joblib"
                if classifier.exists() and regressor.exists():
                    return True

        return False

    def _update_models(self) -> bool:
        """
        Check for and download model updates.

        Returns:
            True if models were updated, False otherwise.
        """
        # Record check time
        with open(self._last_check_file, "w") as f:
            f.write(datetime.now().isoformat())

        # Get remote version
        remote_version = self.get_remote_version()
        if not remote_version:
            logger.info("No remote models available")
            return False

        # Compare with cached version
        cached_version = self.get_cached_version()
        if cached_version == remote_version:
            logger.info(f"Models are up to date (version: {cached_version})")
            return False

        logger.info(f"Downloading models (version: {remote_version})")

        # Download and extract
        if self.storage_type == "github":
            success = self._download_from_github(remote_version)
        elif self.storage_type == "s3":
            success = self._download_from_s3()
        else:
            logger.error(f"Unknown storage type: {self.storage_type}")
            return False

        if success:
            # Update metadata
            self._save_metadata({"version": remote_version})
            logger.info("Models updated successfully")

        return success

    def _get_github_latest_release(self) -> Optional[str]:
        """Get tag of latest GitHub release."""
        url = f"https://api.github.com/repos/{self.repo}/releases/latest"
        headers = self._github_headers()

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 404:
                # No releases yet
                return None
            response.raise_for_status()
            data = response.json()
            return data.get("tag_name")
        except requests.RequestException as e:
            logger.warning(f"Failed to get latest release: {e}")
            return None

    def _download_from_github(self, tag: str) -> bool:
        """Download models from GitHub release."""
        # Get release assets
        url = f"https://api.github.com/repos/{self.repo}/releases/tags/{tag}"
        headers = self._github_headers()

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            release = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get release info: {e}")
            return False

        # Find models asset
        asset = None
        for a in release.get("assets", []):
            if a["name"] == DEFAULT_ASSET_NAME:
                asset = a
                break

        if not asset:
            logger.error(f"No {DEFAULT_ASSET_NAME} found in release {tag}")
            return False

        # Download asset
        download_url = asset["url"]
        headers = self._github_headers()
        headers["Accept"] = "application/octet-stream"

        try:
            response = requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            # Save to temp file and extract
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Extract
            self._extract_models(tmp_path)

            # Cleanup temp file
            os.unlink(tmp_path)

            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download models: {e}")
            return False

    def _get_s3_latest_version(self) -> Optional[str]:
        """Get version of latest S3 models."""
        # This would check S3 for latest version marker
        # For now, return timestamp-based version
        try:
            import boto3

            bucket = os.environ.get(S3_BUCKET_ENV)
            if not bucket:
                return None

            s3 = boto3.client("s3")
            response = s3.head_object(Bucket=bucket, Key="models/version.txt")
            return response.get("ETag", "").strip('"')
        except Exception:
            return None

    def _download_from_s3(self) -> bool:
        """Download models from S3."""
        try:
            import boto3

            bucket = os.environ.get(S3_BUCKET_ENV)
            if not bucket:
                logger.error("S3_BUCKET environment variable not set")
                return False

            s3 = boto3.client("s3")

            # Download models archive
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
                s3.download_fileobj(bucket, "models/models.tar.gz", tmp)
                tmp_path = tmp.name

            # Extract
            self._extract_models(tmp_path)

            # Cleanup
            os.unlink(tmp_path)

            return True

        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def _extract_models(self, archive_path: str) -> None:
        """Extract models from archive."""
        # Clear existing models
        if self._models_dir.exists():
            import shutil
            shutil.rmtree(self._models_dir)

        self._models_dir.mkdir(parents=True)

        # Extract archive
        with tarfile.open(archive_path, "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")

            tar.extractall(path=self._models_dir)

    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata about cached models."""
        metadata["downloaded_at"] = datetime.now().isoformat()
        with open(self._metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _github_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers


# Global instance for convenience
_remote_loader: Optional[RemoteModelLoader] = None


def get_remote_loader() -> RemoteModelLoader:
    """Get or create the global remote model loader instance."""
    global _remote_loader
    if _remote_loader is None:
        _remote_loader = RemoteModelLoader()
    return _remote_loader


def get_models_dir(force_refresh: bool = False) -> Path:
    """Convenience function to get models directory."""
    return get_remote_loader().get_models_dir(force_refresh)


def get_trained_symbols() -> List[str]:
    """Convenience function to get trained symbols."""
    return get_remote_loader().get_trained_symbols()


def load_model_metadata(symbol: str) -> Optional[Dict]:
    """Convenience function to load model metadata."""
    return get_remote_loader().load_model_metadata(symbol)


def is_remote_mode() -> bool:
    """Check if running in remote/cloud mode."""
    # Check for Streamlit Cloud or explicit remote mode
    return (
        os.environ.get("STREAMLIT_SHARING") == "1" or
        os.environ.get("MODEL_STORAGE") in ("github", "s3") or
        os.environ.get("REMOTE_MODELS", "").lower() in ("1", "true", "yes")
    )
