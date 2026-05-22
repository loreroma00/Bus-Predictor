"""Static artifact registry for Observatory-owned domain data.

This module owns durable, shared files that are needed by more than one
runtime path.  It deliberately does not own live process services.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "parquets"


class Artifact(str, Enum):
    """Known durable artifacts shared by training and serving code."""

    CANONICAL_STOP_MAP = "canonical_stop_map"
    CANONICAL_STOP_CONFIG = "canonical_stop_config"
    TRAFFIC_AVERAGES = "traffic_averages"
    GTFS_MD5 = "gtfs_md5"


@dataclass
class StaticArtifacts:
    """Resolve, check, and lazily build static domain artifacts."""

    config: dict[str, Any] | None = None
    cache_strategy: Any = None
    project_root: Path = PROJECT_ROOT
    parquet_dir: Path = PARQUET_DIR

    def path(self, artifact: Artifact | str) -> Path:
        """Return the canonical filesystem path for one artifact."""
        artifact = Artifact(artifact)
        paths = {
            Artifact.CANONICAL_STOP_MAP: self.parquet_dir / "stop_route_map.parquet",
            Artifact.CANONICAL_STOP_CONFIG: self.parquet_dir / "stop_route_config.json",
            Artifact.TRAFFIC_AVERAGES: self.parquet_dir / "traffic_averages.parquet",
            Artifact.GTFS_MD5: self.parquet_dir / "gtfs_md5.json",
        }
        return paths[artifact]

    def is_available(self, artifact: Artifact | str) -> bool:
        """Return whether an artifact is ready for use."""
        artifact = Artifact(artifact)
        return self.path(artifact).exists()

    def ensure(
        self,
        artifact: Artifact | str,
        *,
        build: bool = False,
        force: bool = False,
    ) -> bool:
        """Ensure an artifact exists, optionally building it when absent."""
        artifact = Artifact(artifact)
        if not force and self.is_available(artifact):
            return True
        if not build:
            return False

        builder = self._builder_for(artifact)
        if builder is None:
            return False

        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        builder()
        return self.is_available(artifact)

    def _builder_for(self, artifact: Artifact):
        """Return the lazy builder for buildable artifacts."""
        builders = {
            Artifact.CANONICAL_STOP_MAP: self._build_canonical_stop_map,
        }
        return builders.get(artifact)

    def _build_canonical_stop_map(self):
        """Build the canonical stop map through the dataset pipeline facade."""
        from prepare_dataset import build_canonical_shape_map

        build_canonical_shape_map(
            output_path=self.path(Artifact.CANONICAL_STOP_MAP),
            config_output_path=self.path(Artifact.CANONICAL_STOP_CONFIG),
            traffic_output_path=self.path(Artifact.TRAFFIC_AVERAGES),
        )


DEFAULT_ARTIFACTS = StaticArtifacts()
