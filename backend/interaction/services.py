"""
Services - Background threads for data collection, saving, and uptime.

Subscribes to events from the event bus for start/stop control.
"""

import logging
import threading
import queue
import time
from application.live import data
from application.domain.internal_events import domain_events, DIARY_FINISHED
from persistence import saving_loop, saving_parquet, log_uptime, saving_database
from .events import console_events

COLLECTION_THREAD = None
SAVING_THREAD = None
UPTIME_THREAD = None
WEATHER_THREAD = None
GEOCODING_THREAD = None
TRAFFIC_THREAD = None
GUI_THREAD = None
PREDICTION_THREAD = None
PREDICTION_QUEUE = None
TRAFFIC_ANALYSIS_THREAD = None
TRAFFIC_ANALYSIS_QUEUE = None
VEHICLE_ANALYSIS_THREAD = None
VEHICLE_ANALYSIS_QUEUE = None

UPDATE_TIME = 900
TRAFFIC_UPDATE_TIME = 900  # 15 minutes

# State interface for GUI (set by main.py)
_state_interface = None


def uptime_loop(stop_event):
    logging.info(" > Uptime Logger Started.")
    while not stop_event.is_set():
        log_uptime()
        for _ in range(60):
            if stop_event.is_set():
                break
            time.sleep(1)
    logging.info(" > Uptime Logger Stopped.")


def prediction_worker_loop(work_queue: queue.Queue, stop_event: threading.Event):
    """Worker thread for processing completed diaries."""
    logging.info(" > Prediction Worker Started.")
    while not stop_event.is_set():
        try:
            # Wait for item with timeout to check stop_event
            item = work_queue.get(timeout=1.0)
            
            diary = item.get("diary")
            route_id = item.get("route_id")
            observatory = item.get("observatory")
            
            if diary and route_id and observatory:
                try:
                    vectors = observatory.process_completed_diary(diary, route_id)
                    if vectors:
                        pred_cfg = data.CONFIG.get("prediction", {})
                        conn_str = pred_cfg.get("vector_db_connection")
                        table_name = pred_cfg.get("vector_table")
                        
                        strategy = saving_database(connection_string=conn_str, table_name=table_name)
                        strategy.execute(vectors)
                        logging.info(f"📊 Stored {len(vectors)} vectors (Async Worker)")
                except Exception as e:
                    logging.error(f"⚠️ Prediction Worker Error: {e}")
            
            work_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Prediction Worker Loop Error: {e}")
            
    logging.info(" > Prediction Worker Stopped.")


def traffic_worker_loop(work_queue: queue.Queue, stop_event: threading.Event):
    """Worker thread for processing completed diaries (Traffic Analysis)."""
    logging.info(" > Traffic Analysis Worker Started.")
    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=1.0)
            
            diary = item.get("diary")
            observatory = item.get("observatory")
            
            if diary and observatory:
                try:
                    vectors = observatory.process_traffic_diary(diary)
                    if vectors:
                        traf_cfg = data.CONFIG.get("traffic", {})
                        conn_str = traf_cfg.get("vector_db_connection")
                        table_name = traf_cfg.get("vector_table")
                        
                        strategy = saving_database(connection_string=conn_str, table_name=table_name)
                        strategy.execute(vectors)
                        logging.info(f"🚗 Generated {len(vectors)} traffic vectors")
                except Exception as e:
                    logging.error(f"⚠️ Traffic Worker Error: {e}")
            
            work_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Traffic Worker Loop Error: {e}")
            
    logging.info(" > Traffic Analysis Worker Stopped.")


def vehicle_worker_loop(work_queue: queue.Queue, stop_event: threading.Event):
    """Worker thread for processing completed diaries (Vehicle Classification)."""
    # Import locally to avoid circular dependencies
    from application.post_processing.data_cleaning import VehiclePipeline

    logging.info(" > Vehicle Analysis Worker Started.")
    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=1.0)
            
            diary = item.get("diary")
            observatory = item.get("observatory")
            vehicle_type_name = item.get("vehicle_type_name", "Unknown")
            
            if diary and observatory:
                try:
                    # Instantiate pipeline directly using event data
                    # Ensure ledger is loaded
                    ledger = observatory.get_ledger()
                    
                    pipeline = VehiclePipeline(
                        diary=diary,
                        ledger=ledger,
                        config=observatory.config,
                        vehicle_type_name=vehicle_type_name
                    )
                    
                    vectors = pipeline.clean()
                    
                    if vectors:
                        veh_cfg = data.CONFIG.get("vehicle", {})
                        conn_str = veh_cfg.get("vector_db_connection")
                        table_name = veh_cfg.get("vector_table")
                        
                        strategy = saving_database(connection_string=conn_str, table_name=table_name)
                        strategy.execute(vectors)
                        logging.info(f"🚌 Generated {len(vectors)} vehicle vectors")
                except Exception as e:
                    logging.error(f"⚠️ Vehicle Worker Error: {e}")
            
            work_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Vehicle Worker Loop Error: {e}")
            
    logging.info(" > Vehicle Analysis Worker Stopped.")


def start_services(observatory=None, config: dict = None):
    """Start all background services."""
    global \
        COLLECTION_THREAD, \
        SAVING_THREAD, \
        UPTIME_THREAD, \
        WEATHER_THREAD, \
        GEOCODING_THREAD, \
        TRAFFIC_THREAD, \
        GUI_THREAD, \
        PREDICTION_THREAD, \
        PREDICTION_QUEUE, \
        TRAFFIC_ANALYSIS_THREAD, \
        TRAFFIC_ANALYSIS_QUEUE, \
        VEHICLE_ANALYSIS_THREAD, \
        VEHICLE_ANALYSIS_QUEUE, \
        _state_interface
        
    logging.info("Starting Services...")
    data.STOP_COLLECTION_EVENT.clear()
    
    config = config or {}
    timings = config.get("timings", {})
    services_cfg = config.get("services", {})
    
    update_time = int(timings.get("update_interval", 900))
    traffic_update_time = int(timings.get("traffic_update_interval", 900))
    gui_port = int(services_cfg.get("debug_gui_port", 8050))

    # Initialize Prediction Queue and Worker
    if PREDICTION_QUEUE is None:
        PREDICTION_QUEUE = queue.Queue()
        
    if PREDICTION_THREAD is None or not PREDICTION_THREAD.is_alive():
        PREDICTION_THREAD = threading.Thread(
            target=prediction_worker_loop,
            args=(PREDICTION_QUEUE, data.STOP_COLLECTION_EVENT),
            daemon=True
        )
        PREDICTION_THREAD.start()

    # Initialize Traffic Analysis Pipeline
    if TRAFFIC_ANALYSIS_QUEUE is None:
        TRAFFIC_ANALYSIS_QUEUE = queue.Queue()
        
    if TRAFFIC_ANALYSIS_THREAD is None or not TRAFFIC_ANALYSIS_THREAD.is_alive():
        TRAFFIC_ANALYSIS_THREAD = threading.Thread(
            target=traffic_worker_loop,
            args=(TRAFFIC_ANALYSIS_QUEUE, data.STOP_COLLECTION_EVENT),
            daemon=True
        )
        TRAFFIC_ANALYSIS_THREAD.start()
        
    # Initialize Vehicle Analysis Pipeline
    if VEHICLE_ANALYSIS_QUEUE is None:
        VEHICLE_ANALYSIS_QUEUE = queue.Queue()

    if VEHICLE_ANALYSIS_THREAD is None or not VEHICLE_ANALYSIS_THREAD.is_alive():
        VEHICLE_ANALYSIS_THREAD = threading.Thread(
            target=vehicle_worker_loop,
            args=(VEHICLE_ANALYSIS_QUEUE, data.STOP_COLLECTION_EVENT),
            daemon=True
        )
        VEHICLE_ANALYSIS_THREAD.start()

    # Subscribe to domain events
    domain_events.subscribe(DIARY_FINISHED, _on_diary_finished)

    if COLLECTION_THREAD is None or not COLLECTION_THREAD.is_alive():
        COLLECTION_THREAD = threading.Thread(
            target=data.run_collection_loop,
            daemon=True,
        )
        COLLECTION_THREAD.start()
        logging.info(" > Collection Thread Started.")
    else:
        logging.info(" > Collection Thread already running.")

    if SAVING_THREAD is None or not SAVING_THREAD.is_alive():
        SAVING_THREAD = threading.Thread(
            target=saving_loop,
            args=(
                data.OBSERVATORY,
                saving_parquet(),
                data.STOP_COLLECTION_EVENT,
                data.CACHE_STRATEGY,  # Pass cache strategy for city cache saving
            ),
        )
        SAVING_THREAD.start()
        logging.info(" > Saving Thread Started.")
    else:
        logging.info(" > Saving Thread already running.")

    if UPTIME_THREAD is None or not UPTIME_THREAD.is_alive():
        UPTIME_THREAD = threading.Thread(
            target=uptime_loop, args=(data.STOP_COLLECTION_EVENT,), daemon=True
        )
        UPTIME_THREAD.start()
    else:
        logging.info(" > Uptime Thread already running.")

    if WEATHER_THREAD is None or not WEATHER_THREAD.is_alive():
        WEATHER_THREAD = threading.Thread(
            target=data.run_weather_loop,
            args=(update_time,),
            daemon=True,
        )
        WEATHER_THREAD.start()
    else:
        logging.info(" > Weather Thread already running.")

    # Only start geocoding thread for async mode (rate-limited servers)
    from application.domain.map_info import AsyncGeocodingService

    if data.OBSERVATORY and isinstance(
        data.OBSERVATORY._geocoding, AsyncGeocodingService
    ):
        if GEOCODING_THREAD is None or not GEOCODING_THREAD.is_alive():
            GEOCODING_THREAD = threading.Thread(
                target=data.run_geocoding_loop,
                daemon=True,
            )
            GEOCODING_THREAD.start()
            logging.info(" > Geocoding Thread Started (async mode).")
        else:
            logging.info(" > Geocoding Thread already running.")
    else:
        logging.info(" > Geocoding Thread NOT needed (sync mode).")

    # Start traffic thread if traffic service is configured
    if data.TRAFFIC_SERVICE:
        if TRAFFIC_THREAD is None or not TRAFFIC_THREAD.is_alive():
            TRAFFIC_THREAD = threading.Thread(
                target=data.run_traffic_loop,
                args=(traffic_update_time,),
                daemon=True,
            )
            TRAFFIC_THREAD.start()
            logging.info(" > Traffic Thread Started.")
        else:
            logging.info(" > Traffic Thread already running.")
    else:
        logging.info(" > Traffic Thread NOT started (no API key).")

    # Start Debug GUI if state interface is available
    if _state_interface is not None:
        if GUI_THREAD is None or not GUI_THREAD.is_alive():
            from . import debug_gui

            GUI_THREAD = debug_gui.start_gui(_state_interface, port=gui_port)
        else:
            logging.info(" > Debug GUI already running.")


def set_state_interface(state_interface):
    """Set the state interface for the GUI (called from main.py)."""
    global _state_interface
    _state_interface = state_interface


def stop_services():
    """Stop all background services gracefully."""
    logging.info("Stopping Services (Collection & Auto-Save)...")
    data.STOP_COLLECTION_EVENT.set()
    
    # Unsubscribe
    domain_events.unsubscribe(DIARY_FINISHED, _on_diary_finished)

    # Stop GUI
    try:
        from . import debug_gui

        debug_gui.stop_gui()
    except Exception:
        pass


# ============================================================
# Event Subscriptions
# ============================================================

def _on_diary_finished(data: dict):
    """Handler: Pushes completed diary to prediction and traffic queues."""
    if PREDICTION_QUEUE is not None:
        PREDICTION_QUEUE.put(data)
    
    if TRAFFIC_ANALYSIS_QUEUE is not None:
        TRAFFIC_ANALYSIS_QUEUE.put(data)

    if VEHICLE_ANALYSIS_QUEUE is not None:
        VEHICLE_ANALYSIS_QUEUE.put(data)


def _on_services_start(event_data):
    """Handler for services_start event."""
    start_services()


def _on_services_stop(event_data):
    """Handler for services_stop event."""
    stop_services()


def _on_shutdown(event_data):
    """Handler for shutdown_requested event."""
    data.SHUTDOWN_EVENT.set()
    stop_services()


# Register handlers
console_events.subscribe("services_start", _on_services_start)
console_events.subscribe("services_stop", _on_services_stop)
console_events.subscribe("shutdown_requested", _on_shutdown)
