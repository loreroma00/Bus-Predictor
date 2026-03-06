# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATAC Bus Delay Prediction System — an ML system for predicting bus arrival delays in Rome's ATAC network. Combines real-time GTFS data collection, TomTom traffic data, Open-Meteo weather data, and deep learning models (LSTM/ODE-LSTM) to forecast per-stop delays.

**Tech stack**: Python 3.11+, FastAPI, PyTorch, pandas, TimescaleDB/PostgreSQL, asyncio

## Repository Layout

- **`backend/`** — Main application (domain models, data collection, ML, API server)
- **`frontend-thesis/`** — CLI client for predictions and validation
- **`backend-thesis/`** — Legacy backend (mostly unused)
- **`thesisProjectReal/`** — Original thesis data files

## Running the Application

All commands run from `backend/`. Always use the venv: `source venv/bin/activate`

```bash
# Data collection (GTFS-RT feeds + traffic + weather)
python main.py collect [--debug] [--lenient-pipeline]

# Prediction API server (FastAPI, default 0.0.0.0:8000)
python main.py serve [--model NAME] [--host HOST] [--port PORT]

# Test database connections
python main.py test-db

# ML dataset preparation
python prepare_dataset.py [--skip-db] [--force-canonical] [--start-date YYYY-MM-DD]

# Model training
cd application/model && python train.py

# Frontend CLI (from frontend-thesis/)
python cli.py                              # Interactive prediction
python cli.py --test-model 2026-02-28      # Retrospective validation
python cli.py --live-validate 02-03-2026   # Live validation via WebSocket
```

## Tests

```bash
pytest backend/tests/
```

15 test files covering domain entities, data cleaning pipelines, traffic services, cache strategies, geocoding, events, and integration tests.

## Architecture

**Domain-Driven Design** with clear layer separation:

- **`application/domain/`** — Core entities: Route, Trip, Stop, Bus, Measurement, Diary. The `Observatory` is the main facade coordinating data flow.
- **`application/live/`** — Real-time GTFS-RT feed fetching (protobuf), TomTom traffic fetching.
- **`application/post_processing/`** — Data cleaning pipelines (prediction/traffic/lenient) and feature vectorization.
- **`application/preprocessing/`** — ML dataset preparation: canonical shape mapping (routes to 100 segments), vector processing, feature scaling.
- **`application/model/`** — `BusLSTM` / `BusODELSTM` architectures, training, and inference (`Predictor` class).
- **`application/services/`** — Application services: shared state, live validation, bus type prediction.
- **`persistence/`** — TimescaleDB async connections, file/no-cache strategies, Parquet diary storage.
- **`interaction/`** — CLI console commands, Dash debug GUI, wiring/initialization in `interaction/main.py`.

**Key patterns**:
- **Observer pattern**: `ObserverManager` manages `Diary` objects (one per active trip), each tracking `Measurement` observations. `DIARY_FINISHED` domain event fires on trip completion.
- **Strategy pattern**: `CacheStrategy` (file/none), `TripVerificationStrategy` (basic/scaled), `WeatherStrategy` (greedy, etc.) — all injected into Observatory.
- **Dependency injection**: Observatory and commands receive dependencies via constructor; wiring happens in `interaction/main.py`.

**Data flow**: GTFS-RT feeds → Observatory → Buses → Observers → Diaries → Data Cleaning → Vectorization → TimescaleDB / Parquet storage → ML training → Predictor inference → REST API response.

## Configuration

`backend/config.ini` (copy from `config.ini.example`). Sections: `[api]`, `[prediction]`, `[traffic]`, `[vehicle]` (DB connections), `[urls]` (GTFS/weather/traffic feeds), `[paths]`, `[timings]`, `[services]`, `[data_cleaning]`.

Environment variable overrides: `TOMTOM_API_KEY`, `TIMESCALE_CONNECTION_STRING`.

## Code Style & Conventions

- Avoid warnings in all code.
- Low coupling, high cohesion, top-down imports (never bottom-up).
- Reuse and generalize existing code when possible.
- Use `pip` for package management.
- Parquet files go in `backend/parquets/`, diary data in `backend/diaries/`.
- If anything is unclear, ask before acting. If something can be improved, point it out with reasoning.
