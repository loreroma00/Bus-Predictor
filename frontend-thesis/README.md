# ATAC Bus Delay Prediction - Frontend CLI

Command-line tool for interacting with the ATAC Bus Delay Prediction API.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Prediction

Run without arguments for interactive mode:

```bash
python cli.py
```

You will be prompted to:
1. Select a bus line (e.g., 64, 780, C2)
2. Select a direction (0 or 1)
3. Enter a date (DD-MM-YYYY)
4. Enter a time (HH:MM)
5. Enter bus type (or press Enter for default)

### Retrospective Validation

Validate the model against historical ground truth data for a specific date:

```bash
python cli.py --test-model 2026-02-28
# or
python cli.py --test-model 28-02-2026
```

This compares predictions against measurements stored in the diaries database.

### Live Validation

Live validation monitors real-time data collection and validates predictions as trips complete.

#### Start Live Validation

```bash
python cli.py --live-validate 02-03-2026
```

This will:
1. Schedule a validation session for the date
2. Predict ALL scheduled trips upfront (batched inference)
3. Connect via WebSocket for real-time updates
4. Display validation results as trips complete
5. Auto-stop at 4AM the next day

#### Monitor Progress

The CLI displays real-time updates:

```
[10:45:23] Validated: 12/150 | Pending: 138
[10:46:15] Trip 64_0_08:30: Route 64 Dir 0 | RMSE: 15.2s | Measurements: 18
[10:47:02] Validated: 13/150 | Pending: 137
...
```

#### Check Status

Check status of current/recent session:

```bash
python cli.py --live-status
```

Output:
```
LIVE VALIDATION STATUS
============================================================
Session ID:     a1b2c3d4-e5f6-7890-abcd-ef1234567890
Date:           02-03-2026
Status:         monitoring
Scheduled:      150
Predicted:      148
Validated:      42
Pending:        106
Started at:     2026-03-02T06:00:00
Stops at:       2026-03-03T04:00:00

Delay Metrics:
  Median RMSE:  17.32s
  Min RMSE:     5.21s
  Max RMSE:     89.45s
============================================================
```

#### Stop Session

Stop a running session before auto-timeout:

```bash
python cli.py --live-stop
```

## API Reference

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single trip prediction |
| `/validate` | POST | Retrospective validation |
| `/validate/live/schedule` | POST | Start live validation |
| `/validate/live/stop` | POST | Stop live validation |
| `/validate/live/status` | GET | Get session status |
| `/health` | GET | Health check |

### WebSocket Endpoint

| Endpoint | Description |
|----------|-------------|
| `/validate/live/ws/{session_id}` | Real-time validation updates |

### WebSocket Message Types

**Status Update:**
```json
{
  "type": "status",
  "status": "monitoring",
  "total_scheduled": 150,
  "total_predicted": 148,
  "total_validated": 42,
  "total_pending": 106,
  "timestamp": "2026-03-02T10:45:00"
}
```

**Trip Validated:**
```json
{
  "type": "trip_validated",
  "trip_id": "64_0_08:30",
  "route_id": "64",
  "direction_id": 0,
  "scheduled_start": "08:30:00",
  "mse": 245.3,
  "rmse": 15.7,
  "n_measurements": 18,
  "timestamp": "2026-03-02T10:45:23"
}
```

**Completed:**
```json
{
  "type": "completed",
  "median_mse": 312.5,
  "median_rmse": 17.7,
  "total_validated": 142,
  "log_file": "validation_live_20260302.log",
  "report_file": "validation_live_20260302_report.txt",
  "timestamp": "2026-03-02T23:59:00"
}
```

## Output Files

### Live Validation

Results are saved to `backend/results/`:

- `validation_live_{YYYYMMDD}.log` - Per-trip validation details
- `validation_live_{YYYYMMDD}_report.txt` - Summary report with confusion matrix

### Retrospective Validation

Results are saved to `backend/results/`:

- `validation_{YYYYMMDD}.log` - Per-trip validation details  
- `validation_{YYYYMMDD}_report.txt` - Summary report

## Architecture

```
┌─────────────────┐     HTTP/WS      ┌─────────────────┐
│   Frontend CLI  │ ◄──────────────► │   Backend API   │
│                 │                  │                 │
│  - cli.py       │                  │  - main.py      │
│  - api_client.py│                  │  - validator.py │
└─────────────────┘                  └─────────────────┘
                                            │
                                            ▼
                                     ┌─────────────────┐
                                     │  ObserverManager│
                                     │  (Diaries)      │
                                     └─────────────────┘
```

## Live Validation Flow

```
1. POST /validate/live/schedule {"date": "DD-MM-YYYY"}
   │
   ├─► Load scheduled trips from GTFS ledger
   │
   ├─► Run batch predictions (all trips upfront)
   │
   └─► Subscribe to DIARY_FINISHED events
        │
        ▼
2. WebSocket: /validate/live/ws/{session_id}
   │
   ├─► On each diary finished:
   │    ├─ Match trip_id to prediction
   │    ├─ Compute MSE, RMSE, confusion
   │    └─ Broadcast update via WebSocket
   │
   └─► Stop conditions:
        ├─ All trips validated
        ├─ 4AM next day (timeout)
        └─ User calls /validate/live/stop
        │
        ▼
3. Write output files to results/
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Session already running | Returns 409 Conflict with session info |
| No ground truth data | Trip skipped, logged as warning |
| Prediction fails | Trip added to failed list |
| WebSocket disconnects | Session continues, can reconnect |
| Server restarts | Session lost, must restart |

## Example: Programmatic Usage

```python
from api_client import APIClient, LiveValidationClient
import asyncio

# HTTP client
api = APIClient("http://localhost:8000")

# Start live validation
session = api.validate_live_schedule("02-03-2026")
session_id = session["session_id"]

# WebSocket client
live = LiveValidationClient("http://localhost:8000")

live.on_trip_validated(lambda d: print(f"Trip {d['trip_id']}: RMSE={d['rmse']:.1f}s"))
live.on_completed(lambda d: print(f"Done! Median RMSE: {d['median_rmse']:.1f}s"))

asyncio.run(live.connect(session_id))
```

## Configuration

| Environment | API URL |
|-------------|---------|
| Production | `https://atacapi.loreromaphotos.it` |
| Local | `http://localhost:8000` |

Override with `--api-url`:
```bash
python cli.py --api-url http://localhost:8000 --live-validate 02-03-2026
```
