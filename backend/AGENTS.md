# AGENTS.md

## Setup & Development info:
- We are developing the backend to an app that will be used to interrogate an AI model for bus timing predictions in Rome (ATAC). We are building the ML model itself, too.
- Whenever you need to execute any command, always ensure you're utilising the virtual environment at ./venv; to activate it: source ./venv/bin/activate
- Always use pip unless there's a sound reason not to
- Parquet files go in ./parquets
- Diaries (collected trip data) go in ./diaries

## Project Structure:
- `main.py` - Unified CLI entry point (see below)
- `application/domain/` - Core domain models (routes, trips, vehicles, weather)
- `application/live/` - GTFS real-time feed fetchers, traffic services
- `application/post_processing/` - Data cleaning, vectorization
- `application/preprocessing/` - Shape mapping, vector processing for ML
- `application/model/` - ML model, training, prediction
- `persistence/` - Database connections, caching, diaries storage
- `interaction/` - CLI console, debug GUI, event handling
- `tests/` - Unit and integration tests

## CLI Commands:
- `python main.py collect [--debug] [--lenient-pipeline]` - Run GTFS data collection
- `python main.py serve [--model NAME] [--host HOST] [--port PORT]` - Start FastAPI prediction server
- `python main.py test-db` - Test database connections

## Quality & Style
- Always ensure to avoid warnings
- Always follow best practices; in particular, low coupling, high cohesion, and top-down imports (never bottom-up).
- Always reuse code whenever possible and generalize existing code if possible.

## Interaction rules
- I will always try to give you as precise and accurate guidance as possible, but if anything is unclear to you, MUST ask before acting.
- I will always try to give you as good and efficient implementations as possible, but if anything can be improved, you MUST point it out and explain which way would be better and why.
