"""Model discovery, contract checks, loading, and hotswap support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .model import BusLSTM, OccupancyLSTM
except ImportError:  # pragma: no cover - keeps script-style imports working
    from model import BusLSTM, OccupancyLSTM


MODEL_DIR = Path(__file__).resolve().parent
REQUIRED_CONFIG_KEYS = {
    "x1_dense_features",
    "x2_dense_features",
    "x1_cat_cards",
    "x2_cat_cards",
    "encoder_hidden_size",
    "decoder_hidden_size",
}


@dataclass(frozen=True)
class ModelCandidate:
    """A complete TIME+CROWD checkpoint set that can be loaded."""

    name: str
    time_path: Path
    crowd_path: Path
    config_path: Path

    @property
    def time_filename(self) -> str:
        return self.time_path.name

    @property
    def crowd_filename(self) -> str:
        return self.crowd_path.name

    @property
    def config_filename(self) -> str:
        return self.config_path.name


class LoadedModel:
    """Loaded neural models plus the low-level tensor inference API."""

    def __init__(
        self,
        candidate: ModelCandidate,
        config: dict[str, Any],
        model_time: BusLSTM,
        model_crowd: OccupancyLSTM,
    ):
        self.candidate = candidate
        self.name = candidate.name
        self.config = config
        self.max_stops = int(config["max_stops"])
        self.model_time = model_time
        self.model_crowd = model_crowd

    def _sanitize_categorical_inputs(
        self,
        x1_cat_batch: np.ndarray,
        x2_cat_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp out-of-range categorical indices before embedding lookup."""
        x1_cards = self.config["x1_cat_cards"]
        x2_cards = self.config["x2_cat_cards"]

        for col_idx, card in enumerate(x1_cards):
            if card <= 0:
                continue
            invalid = (x1_cat_batch[:, col_idx] < 0) | (
                x1_cat_batch[:, col_idx] >= card
            )
            if int(invalid.sum()) > 0:
                x1_cat_batch[invalid, col_idx] = 0

        for col_idx, card in enumerate(x2_cards):
            if card <= 0:
                continue
            invalid = (x2_cat_batch[:, :, col_idx] < 0) | (
                x2_cat_batch[:, :, col_idx] >= card
            )
            if int(invalid.sum()) > 0:
                x2_cat_batch[:, :, col_idx][invalid] = 0

        return x1_cat_batch, x2_cat_batch

    def infer_single(
        self,
        x1_cat: np.ndarray,
        x1_dense: np.ndarray,
        x2_cat: np.ndarray,
        x2_dense: np.ndarray,
        t_grid: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run delay and crowd inference for one trip."""
        ms = self.max_stops
        delays, crowd = self.infer_batch(
            x1_cat.reshape(1, -1),
            x1_dense.reshape(1, -1),
            x2_cat.reshape(1, ms, -1),
            x2_dense.reshape(1, ms, -1),
            t_grid.reshape(1, ms),
            lengths,
        )
        return delays.squeeze(0), crowd.squeeze(0)

    def infer_batch(
        self,
        x1_cat_batch: np.ndarray,
        x1_dense_batch: np.ndarray,
        x2_cat_batch: np.ndarray,
        x2_dense_batch: np.ndarray,
        t_grid_batch: np.ndarray,
        lengths_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run delay and crowd inference for a batch of already-built tensors."""
        x1_cat_batch, x2_cat_batch = self._sanitize_categorical_inputs(
            x1_cat_batch,
            x2_cat_batch,
        )

        x1_cat_tensor = torch.tensor(x1_cat_batch, dtype=torch.int64)
        x1_dense_tensor = torch.tensor(x1_dense_batch, dtype=torch.float32)
        x2_cat_tensor = torch.tensor(x2_cat_batch, dtype=torch.int64)
        x2_dense_tensor = torch.tensor(x2_dense_batch, dtype=torch.float32)
        lengths_tensor = torch.tensor(lengths_batch, dtype=torch.int64)
        t_grid_tensor = torch.tensor(t_grid_batch, dtype=torch.float32)

        with torch.no_grad():
            pred_time_scaled = self.model_time(
                x1_cat_tensor,
                x1_dense_tensor,
                x2_cat_tensor,
                x2_dense_tensor,
                lengths=lengths_tensor,
                t_grid=t_grid_tensor,
            )
            pred_crowd_logits = self.model_crowd(
                x1_cat_tensor,
                x1_dense_tensor,
                x2_cat_tensor,
                x2_dense_tensor,
                lengths=lengths_tensor,
            )

        delays = pred_time_scaled.squeeze(-1).numpy() * 600.0
        crowd = pred_crowd_logits.argmax(dim=-1).numpy()
        return delays, crowd


class ModelLoader:
    """Find complete model sets, load one, and support atomic hotswaps."""

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.current: LoadedModel | None = None

    def discover_models(self) -> list[ModelCandidate]:
        """Return complete `bus_model_TIME_*` + `bus_model_CROWD_*` sets."""
        candidates: list[ModelCandidate] = []
        pattern = self.model_dir / "bus_model_TIME_*.pth"
        for time_path in sorted(self.model_dir.glob(pattern.name)):
            exp_id = time_path.name.replace("bus_model_TIME_", "").replace(".pth", "")
            crowd_path = self.model_dir / f"bus_model_CROWD_{exp_id}.pth"
            config_path = self.model_dir / f"hyperparameters_DUAL_{exp_id}.json"
            if crowd_path.exists() and config_path.exists():
                candidates.append(
                    ModelCandidate(
                        name=exp_id,
                        time_path=time_path,
                        crowd_path=crowd_path,
                        config_path=config_path,
                    )
                )
        return candidates

    def find_incomplete_models(self) -> list[tuple[str, list[str]]]:
        """Return TIME checkpoints whose paired CROWD/config files are missing."""
        incomplete: list[tuple[str, list[str]]] = []
        for time_path in sorted(self.model_dir.glob("bus_model_TIME_*.pth")):
            exp_id = time_path.name.replace("bus_model_TIME_", "").replace(".pth", "")
            missing = []
            crowd_filename = f"bus_model_CROWD_{exp_id}.pth"
            config_filename = f"hyperparameters_DUAL_{exp_id}.json"
            if not (self.model_dir / crowd_filename).exists():
                missing.append(crowd_filename)
            if not (self.model_dir / config_filename).exists():
                missing.append(config_filename)
            if missing:
                incomplete.append((time_path.name, missing))
        return incomplete

    def load_by_exp_id(self, exp_id: str) -> LoadedModel:
        """Load a model pair by experiment id."""
        return self.load_pair(
            f"bus_model_TIME_{exp_id}.pth",
            f"bus_model_CROWD_{exp_id}.pth",
        )

    def load_pair(self, time_filename: str, crowd_filename: str) -> LoadedModel:
        """Load a specific TIME+CROWD pair after validating its contract."""
        candidate = self._candidate_from_filenames(time_filename, crowd_filename)
        loaded = self._load_candidate(candidate)
        self.current = loaded
        return loaded

    def load_from_paths(
        self,
        config_path: str | Path,
        time_weights_path: str | Path,
        crowd_weights_path: str | Path,
        name: str | None = None,
    ) -> LoadedModel:
        """Load an explicit config + TIME + CROWD set."""
        time_path = Path(time_weights_path)
        crowd_path = Path(crowd_weights_path)
        config_path = Path(config_path)
        candidate = ModelCandidate(
            name=name or config_path.stem.replace("hyperparameters_DUAL_", ""),
            time_path=time_path,
            crowd_path=crowd_path,
            config_path=config_path,
        )
        for path, label in (
            (time_path, "TIME model"),
            (crowd_path, "CROWD model"),
            (config_path, "Config file"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        loaded = self._load_candidate(candidate)
        self.current = loaded
        return loaded

    def hotswap(
        self,
        exp_id: str | None = None,
        time_filename: str | None = None,
        crowd_filename: str | None = None,
    ) -> LoadedModel:
        """Load a new model and publish it only after all checks pass."""
        if exp_id:
            loaded = self.load_by_exp_id(exp_id)
        elif time_filename and crowd_filename:
            loaded = self.load_pair(time_filename, crowd_filename)
        else:
            raise ValueError("Provide exp_id or both time_filename and crowd_filename")
        self.current = loaded
        return loaded

    def _candidate_from_filenames(
        self,
        time_filename: str,
        crowd_filename: str,
    ) -> ModelCandidate:
        time_path = self.model_dir / os.path.basename(time_filename)
        crowd_path = self.model_dir / os.path.basename(crowd_filename)
        if not time_path.exists():
            raise FileNotFoundError(f"TIME model not found: {time_path.name}")
        if not crowd_path.exists():
            raise FileNotFoundError(f"CROWD model not found: {crowd_path.name}")

        exp_id = time_path.name.replace("bus_model_TIME_", "").replace(".pth", "")
        config_path = self.model_dir / f"hyperparameters_DUAL_{exp_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path.name}")

        expected_crowd = f"bus_model_CROWD_{exp_id}.pth"
        if crowd_path.name != expected_crowd:
            raise ValueError(
                f"CROWD model {crowd_path.name} does not match TIME experiment {exp_id}"
            )

        return ModelCandidate(
            name=exp_id,
            time_path=time_path,
            crowd_path=crowd_path,
            config_path=config_path,
        )

    def _load_candidate(self, candidate: ModelCandidate) -> LoadedModel:
        with candidate.config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        self._validate_config(config, candidate.config_path.name)

        model_kwargs = {
            "n_x1_dense_features": config["x1_dense_features"],
            "n_x2_dense_features": config["x2_dense_features"],
            "x1_cat_cardinalities": config["x1_cat_cards"],
            "x2_cat_cardinalities": config["x2_cat_cards"],
            "encoder_hidden_size": config["encoder_hidden_size"],
            "lstm_hidden_size": config["decoder_hidden_size"],
        }
        crowd_kwargs = model_kwargs.copy()
        crowd_kwargs["num_lstm_layers"] = config.get("num_lstm_layers", 2)

        model_time = BusLSTM(**model_kwargs)
        model_crowd = OccupancyLSTM(**crowd_kwargs)

        state_time = self._load_state_dict(candidate.time_path)
        state_crowd = self._load_state_dict(candidate.crowd_path)
        model_time.load_state_dict(state_time)
        model_crowd.load_state_dict(state_crowd)
        model_time.eval()
        model_crowd.eval()

        return LoadedModel(candidate, config, model_time, model_crowd)

    def _validate_config(self, config: dict[str, Any], filename: str):
        missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
        if missing:
            raise ValueError(f"{filename} missing required keys: {', '.join(missing)}")
        if "max_stops" not in config:
            raise ValueError(f"{filename} missing required key: max_stops")

    def _load_state_dict(self, path: Path) -> dict[str, Any]:
        state_dict = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        return state_dict
