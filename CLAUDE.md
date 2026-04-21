# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATAC Bus Delay Prediction system for Rome's public transit. Predicts per-stop delay **and** occupancy for scheduled trips using twin LSTM/Neural-ODE models trained on GTFS real-time, TomTom traffic, and Open-Meteo weather data. Two components: a Python `backend/` (collector + FastAPI prediction server + debug GUI) and a Python `frontend-thesis/` CLI.

## Environment

Always activate the root-level venv before running anything: `source venv/bin/activate` (never `pip install` globally — use this venv only). The `backend/` is the default working directory for most commands; `frontend-thesis/` for CLI commands. Both share the root `requirements.txt`.

`backend/config.ini` (copy from `config.ini.example`) is required and holds DB credentials, API keys, URLs, cache strategy, and timing intervals. Env vars `TOMTOM_API_KEY` and `TIMESCALE_CONNECTION_STRING` override config values. Config is parsed into typed dataclasses in `config.py` (`API`, `URLs`, `Paths`, `Timings`, `Services`, `DataCleaning`, `Prediction`, `Traffic`, `Vehicle`).

## Common Commands

```bash
# -- Backend (run from backend/) -----------------------------------

# Real-time data collection + optional debug GUI (http://localhost:8050)
python main.py collect [--debug] [--lenient-pipeline]

# Prediction API server — REQUIRES BOTH models (time + crowd)
python main.py serve --time-model bus_model_mse_ODE.pth \
                     --crowd-model "bus_model_mse_99 LSTM.pth"

# Ping the three TimescaleDB tables configured in config.ini
python main.py test-db

# Build training dataset (4-stage pipeline, see Architecture)
python prepare_dataset.py [--skip-db] [--force-canonical] [--start-date YYYY-MM-DD]

# Train (from backend/application/model/)
python train.py

# Tests
python -m pytest tests/                                      # all
python -m pytest tests/test_domain.py                        # one file
python -m pytest tests/test_domain.py -k "test_name"         # one test

# -- Frontend (run from frontend-thesis/) --------------------------

python cli.py                            # interactive single-trip prediction
python cli.py --test-model 28-02-2026    # retrospective validation vs diaries
python cli.py --live-validate 02-03-2026 # live validation (WebSocket, auto-stops 04:00 next day)
python cli.py --live-status              # inspect running session
python cli.py --live-stop                # stop running session
python cli.py --api-url http://localhost:8000 ...   # target local backend
```

## Architecture

### Dual-model prediction (`application/model/`)

One trip is run through **two independent networks**, chosen at `serve` time via separate flags:

- **BusLSTM** (`model.py`) — delay regression, output scaled by `600.0` (10 minutes). Uses a Neural ODE block (`torchdiffeq`) followed by an LSTM decoder.
- **OccupancyLSTM** (`model.py`) — 7-class crowd classification. Same encoder-decoder shape, different head.

Both share the `x1` / `x2` split:
- **x1 (trip-level)**: route_id, direction_id, day_type, weather_code, bus_type, time_sin/cos
- **x2 (segment-level, 100 steps)**: H3 hex index, stop_sequence, shape_dist, distance_to_next_stop, segment_idx, speed, speed_ratio

Every trip is represented as exactly **`BATCH_SIZE = 100` canonical segments** — the canonical-shape mapper is what makes per-segment predictions comparable across routes. `bus_type` is not user-supplied at predict time; a separate **LightGBM** model (`services/bus_type_predictor.py`, weights `bus_type_predictor.pkl`) infers it from route/time/day features inside the `Predictor`.

`predictor.py` glues encoders (`route_encoder.pkl` / `h3_encoding.json`), both `.pth` checkpoints, and the static stop map (`stop_route_map.parquet`) into a single call that returns `TripForecast` (a list of `StopPrediction`).

### Dataset pipeline (`prepare_dataset.py`, 4 stages)

1. `preprocessing/canonical_shape_mapper.py` — resamples GTFS static shapes into 100 uniform segments per route → `canonical_route_map.parquet`.
2. `application/preprocessing/preprocessing.py` — extracts rows from TimescaleDB into daily `dataset_YYYY-MM-DD.parquet`.
3. `application/preprocessing/vector_processing.py` — aligns raw measurements to canonical segments → `dataset_lstm_unscaled.parquet`.
4. `application/model/scaling_2.py` — scales features and writes final `dataset_lstm_final.parquet` plus the encoders (`route_encoder.pkl`, `h3_encoding.json`, etc.).

`--skip-db` skips stage 2 and reuses daily parquets; `--force-canonical` rebuilds stage 1; `--start-date` filters stages 2+ onwards.

### Real-time collection (`main.py collect`)

Observer pattern rooted in `application/domain/`:

- `ObserverManager` spawns per-vehicle `Observer` objects; each writes a `Diary` (trip-lifetime measurement log) persisted as parquet under `diaries/`.
- `application/live/feed_fetcher.py` polls GTFS-RT protobufs (`gtfs_realtime_pb2.py`) for positions and trip updates.
- `application/live/traffic_service.py` + `traffic_fetcher.py` pull TomTom flow tiles and project them onto H3 hexagons.
- `application/services/shared_state.py` holds the process-wide `WeatherService`, `TrafficService`, ledgers, and observer manager; `weather_strategy` in config picks between `greedy` (all hexagons per cycle) and `subset` (rotate N subsets).
- `application/services/live_validator.py` + `validator.py` subscribe to `DIARY_FINISHED` events, batch-predict all scheduled trips upfront, and broadcast RMSE updates over the `/validate/live/ws/{session_id}` WebSocket.

### Persistence (`persistence/`)

- `database.py` — async `TimescaleDBConnection` wrapper (uses `asyncpg`) for the three tables: transit_vectors, prediction_vectors, traffic_vectors. `test-db` exercises all three.
- `ledger_db.py`, `ledgers/` — GTFS static ledgers (routes, shapes, stops, calendar) materialised once and cached.
- Cache files at repo root: huge `ledger_cache.pkl` (~2 GB) and `city_cache.pkl`. Cache strategy is pluggable via `[services] cache_strategy` (`file` or `none`); see `persistence/strategy.py`.

### API server (`main.py serve`)

FastAPI app assembled inside `main.py` itself. `serve` also starts the collector in the same process so diaries can feed live validation. Key endpoints: `GET /health`, `GET /models`, `GET /routes`, `GET /routes/{route_id}/directions/{dir_id}`, `GET /weather`, `POST /predict`, `POST /validate/live/schedule`, `POST /validate/live/stop`, `GET /validate/live/status`, `WS /validate/live/ws/{session_id}`. Validation results go to `backend/results/validation_live_*.log|_report.txt`.

### Debug GUI (`interaction/debug_gui.py`)

Dash + Dash-Leaflet board on `http://localhost:{debug_gui_port}` (default `8050`) that visualises live ledgers, hex overlays, traffic, and weather while `collect` runs. Slash-command buttons in `interaction/commands.py` and `console.py` drive the same actions as the CLI but scoped to the running process.

## Spatial & schedule conventions

- **H3 (Uber)** hexagons encode location everywhere — traffic tiles, weather subsets, model `x2` features, map overlays. Resolution is configured in `config.py` / domain helpers, not hard-coded in call sites.
- All routes are normalised to **100 segments** (`BATCH_SIZE`). Per-stop predictions are produced by mapping stops onto segments in `predictor.py`; don't change one without the other.
- Dates are `DD-MM-YYYY` on the CLI/API boundary but `YYYY-MM-DD` inside parquet filenames — the frontend CLI accepts either and normalises.
