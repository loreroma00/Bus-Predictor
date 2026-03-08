# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATAC Bus Delay Prediction system for Rome's public transit (ATAC). Predicts bus delays and occupancy using LSTM/ODE neural networks trained on GTFS real-time data, traffic, and weather. Two main components: a Python backend (data collection + FastAPI prediction server) and a Python CLI frontend.

## Common Commands

All commands run from `backend/` unless noted. Activate venv first: `source venv/bin/activate` (root-level venv).

```bash
# Data collection pipeline
python main.py collect [--debug] [--lenient-pipeline]

# Start prediction API server
python main.py serve --model bus_model_mse_ODE.pth

# Test database connections
python main.py test-db

# Run tests
cd backend && python -m pytest tests/
python -m pytest tests/test_domain.py        # single test file
python -m pytest tests/test_domain.py -k "test_name"  # single test

# Dataset preparation (from backend/)
python prepare_dataset.py [--skip-db] [--start-date YYYY-MM-DD] [--force-canonical]

# Model training (from backend/application/model/)
python train.py

# Frontend CLI (from frontend-thesis/)
python cli.py                          # interactive prediction
python cli.py --test-model 28-02-2026  # retrospective validation
python cli.py --live-validate 02-03-2026  # live validation via WebSocket
```

## Architecture

### Dual-Model Prediction System

The predictor (`application/model/predictor.py`) uses **two separate models** for a single trip:
- **BusLSTM** (`model.py`) — predicts delay (time). Uses Neural ODE + LSTM. Output scaled by 600.0 (10 minutes).
- **OccupancyLSTM** (`model.py`) — predicts occupancy/crowd level (classification, 7 classes).

Both share the same encoder-decoder architecture (categorical embeddings → FCNN encoder → LSTM decoder) but have different output heads. Each trip is represented as 100 fixed segments (`BATCH_SIZE = 100`).

Model inputs are split into:
- **x1** (trip-level): route_id, direction_id, day_type, weather_code, bus_type, time_sin/cos
- **x2** (segment-level, 100 steps): H3 hex index, stop_sequence, shape_dist, distance_to_next_stop, segment_idx, speed, speed_ratio

A **LightGBM classifier** (`model_bus.py`) separately predicts bus_type from route/time/day features.

### Data Pipeline (4 stages)

1. `canonical_shape_mapper` — Maps GTFS static shapes to 100 uniform segments per route
2. `preprocessing` — Extracts raw data from TimescaleDB into daily parquet files
3. `vector_processing` — Converts to LSTM-ready vectors
4. `scaling_2` — Feature scaling → `dataset_lstm_final.parquet`

### Real-Time Collection (`main.py collect`)

Uses an observer pattern to track buses in real-time:
- **Observatory** / **ObserverManager** (`domain/`) — manages per-vehicle Observers that record Diaries (trip measurement logs)
- **FeedFetcher** (`live/`) — polls GTFS-RT protobuf feeds for vehicle positions and trip updates
- **TrafficService** (`live/`) — fetches TomTom traffic flow tiles, mapped to H3 hexagons
- **WeatherService** (`services/`) — Open-Meteo weather data, with configurable update strategy (greedy/subset)
- **Diaries** are persisted as parquet files in `diaries/`

### Persistence Layer

- **TimescaleDB** (PostgreSQL) — stores transit vectors, prediction vectors, traffic vectors
- **File-based caching** — `ledger_cache.pkl`, `city_cache.pkl` for GTFS static data
- Configurable cache strategy via `config.ini` (`[services] cache_strategy`)

### API Server (`main.py serve`)

FastAPI app with endpoints for prediction, routes, weather, and live validation. Key endpoints:
- `POST /predict` — single trip delay prediction
- `POST /validate/live/schedule` — start live validation session
- `WS /validate/live/ws/{session_id}` — WebSocket for real-time validation updates

### Frontend CLI

`frontend-thesis/cli.py` — interactive prediction client and validation tool. Connects to backend via HTTP (`api_client.py`) and WebSocket for live validation.

## Configuration

`backend/config.ini` (copied from `config.ini.example`). Environment variables override config values:
- `TOMTOM_API_KEY` — TomTom traffic API key
- `TIMESCALE_CONNECTION_STRING` — PostgreSQL connection string

Config is loaded via `config.py` which exposes typed classes: `API`, `URLs`, `Paths`, `Timings`, `Services`, `DataCleaning`, `Prediction`, `Traffic`, `Vehicle`.

## Key Spatial Concepts

- **H3 hexagons** — Uber's H3 spatial indexing used throughout for location encoding
- Routes are normalized to 100 segments via canonical shape mapping
- Stop sequences are mapped to segments; predictions are per-segment, then aggregated to stops

## Dependencies

Root `requirements.txt` covers both components. Key: PyTorch, torchdiffeq (Neural ODE), LightGBM, FastAPI, asyncpg, h3, geopandas, scikit-learn, pandas.
