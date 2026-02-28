# ATAC Bus Delay Prediction Backend

Unified backend for GTFS real-time data collection and bus delay prediction.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure
cp config.ini.example config.ini
# Edit config.ini with your API keys and database credentials

# Run data collection
python main.py collect

# Run prediction server
python main.py serve --model bus_model_mse_0.pth
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py collect` | Run GTFS real-time data collection pipeline |
| `python main.py serve` | Start FastAPI prediction server |
| `python main.py test-db` | Test database connections |

### Collect Options
- `--debug` - Enable verbose logging
- `--lenient-pipeline` - Use lenient data cleaning

### Serve Options
- `--model NAME` - Model filename to load (e.g., `bus_model_mse_0.pth`)
- `--host HOST` - Host to bind (default: from config)
- `--port PORT` - Port to bind (default: from config)

## Project Structure

```
backend/
├── main.py                 # Unified CLI entry point
├── config.py               # Configuration loader
├── application/
│   ├── domain/             # Core domain models
│   ├── live/               # GTFS feed fetchers, traffic services
│   ├── post_processing/    # Data cleaning, vectorization
│   ├── preprocessing/      # Shape mapping for ML
│   ├── model/              # ML model, training, prediction
│   └── services/           # Shared state, weather service
├── persistence/            # Database, caching, storage
├── interaction/            # CLI console, debug GUI
└── tests/                  # Unit and integration tests
```

## API Endpoints

When running `python main.py serve`:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/models` | GET | List available trained models |

### Weather

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/weather` | GET | Get current weather for Rome |
| `/weather?lat=41.9&lon=12.5` | GET | Get weather for coordinates |
| `/weather?hex_id=8a123...` | GET | Get weather for hexagon |

### GTFS Static Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/routes` | GET | List all routes |
| `/routes/{route_id}/directions` | GET | List directions for a route |
| `/routes/{route_id}/directions/{dir_id}` | GET | Full info: shape, stops, schedule |

### Prediction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict delays with full stop info |

#### Predict Request

```json
{
  "route_id": "62",
  "direction_id": 0,
  "start_date": "28-02-2026",
  "start_time": "07:30",
  "weather_code": 1,
  "bus_type": 0
}
```

#### Predict Response

```json
{
  "route_id": "62",
  "direction_id": 0,
  "trip_date": "28-02-2026",
  "scheduled_start": "07:30:00",
  "weather_code": 1,
  "bus_type": 0,
  "stop_sequence": {
    "1": {
      "stop_sequence": 1,
      "stop_id": "74761",
      "stop_name": "TERMINI",
      "stop_lat": 41.901,
      "stop_lon": 12.500,
      "predicted_arrival": "07:30:45",
      "delay_seconds": 45.0,
      "confidence_rating": null
    }
  }
}
```

## Configuration

Copy `config.ini.example` to `config.ini` and configure:

- TomTom API key for traffic data
- Database connection strings
- GTFS feed URLs
- API server settings

## ML Training Pipeline

### Prepare Dataset

Run the full preprocessing pipeline to create training data:

```bash
python prepare_dataset.py
```

Options:
- `--skip-db` - Skip database extraction (use existing `dataset_*.parquet` files)
- `--force-canonical` - Force regeneration of canonical route map
- `--start-date YYYY-MM-DD` - Only process data from this date onwards

Examples:
```bash
# Full pipeline
python prepare_dataset.py

# Only data from January 1st, 2025 onwards
python prepare_dataset.py --start-date 2025-01-01

# Skip DB, use existing parquet files
python prepare_dataset.py --skip-db

# Force regenerate canonical map + filter by date
python prepare_dataset.py --force-canonical --start-date 2025-01-01
```

### Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1 | `canonical_shape_mapper` | GTFS static files | `canonical_route_map.parquet` |
| 2 | `preprocessing` | TimescaleDB | `dataset_YYYY-MM-DD.parquet` |
| 3 | `vector_processing` | Daily datasets | `dataset_lstm_unscaled.parquet` |
| 4 | `scaling_2` | Unscaled dataset | `dataset_lstm_final.parquet` |

### Output Files

After running `prepare_dataset.py`:

```
parquets/
├── canonical_route_map.parquet   # Route shapes mapped to 100 segments
├── dataset_lstm_final.parquet    # Final training dataset
├── route_encoder.pkl             # Route ID label encoder
├── route_encoding.json           # Route ID to index mapping
├── h3_encoding.json              # H3 hexagon to index mapping
└── traffic_averages.parquet      # Traffic patterns by hex/hour (optional)
```

### Training

After preparing the dataset, train the model:

```bash
cd application/model
python train.py
```
