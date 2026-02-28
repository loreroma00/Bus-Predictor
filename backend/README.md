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
│   └── model/              # ML model, training, prediction
├── persistence/            # Database, caching, storage
├── interaction/            # CLI console, debug GUI
└── tests/                  # Unit and integration tests
```

## API Endpoints

When running `python main.py serve`:

- `GET /models` - List available trained models
- `POST /predict` - Predict bus delays for a route
- `GET /health` - Health check

## Configuration

Copy `config.ini.example` to `config.ini` and configure:

- TomTom API key for traffic data
- Database connection strings
- GTFS feed URLs
- API server settings
