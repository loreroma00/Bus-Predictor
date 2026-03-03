"""
API client for the ATAC Bus Delay Prediction backend.
"""

import json
import requests
from typing import Optional, Callable

import websockets

API_URL = "https://atacapi.loreromaphotos.it"


class APIClient:
    """HTTP client for the prediction API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or API_URL).rstrip("/")
        self.ws_url = self.base_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )

    def predict(
        self,
        route_id: str,
        direction_id: int,
        start_date: str,
        start_time: str,
        weather_code: int,
        bus_type: int,
    ) -> dict:
        """
        Request a trip prediction from the API.

        Args:
            route_id: Bus line identifier (e.g., "211")
            direction_id: 0 or 1
            start_date: DD-MM-YYYY format
            start_time: HH:MM format
            weather_code: WMO weather code
            bus_type: Bus type identifier

        Returns:
            Prediction response dict with stops and delays
        """
        url = f"{self.base_url}/predict"
        payload = {
            "route_id": route_id,
            "direction_id": direction_id,
            "start_date": start_date,
            "start_time": start_time,
            "weather_code": weather_code,
            "bus_type": bus_type,
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def validate(self, date: str) -> dict:
        """
        Request model validation for a specific date (retrospective).

        Args:
            date: DD-MM-YYYY format

        Returns:
            Validation response dict with error metrics and trip summaries
        """
        url = f"{self.base_url}/validate"
        payload = {"date": date}
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def validate_live_schedule(self, date: str) -> dict:
        """
        Schedule and start a live validation session for a date.

        Args:
            date: DD-MM-YYYY format

        Returns:
            Dict with session_id, status, total_scheduled, etc.
        """
        url = f"{self.base_url}/validate/live/schedule"
        payload = {"date": date}
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def validate_live_stop(self) -> dict:
        """
        Stop the current live validation session.

        Returns:
            Dict with session_id and status
        """
        url = f"{self.base_url}/validate/live/stop"
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def validate_live_status(self) -> dict:
        """
        Get the current live validation session status.

        Returns:
            Dict with session status, or empty dict if no session
        """
        url = f"{self.base_url}/validate/live/status"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class LiveValidationClient:
    """WebSocket client for live validation updates."""

    def __init__(self, base_url: str):
        self.ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.session_id: Optional[str] = None
        self._on_status: Optional[Callable] = None
        self._on_progress: Optional[Callable] = None
        self._on_trip_validated: Optional[Callable] = None
        self._on_completed: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def on_status(self, callback: Callable):
        """Register callback for status updates."""
        self._on_status = callback

    def on_progress(self, callback: Callable):
        """Register callback for progress updates."""
        self._on_progress = callback

    def on_trip_validated(self, callback: Callable):
        """Register callback for trip validation events."""
        self._on_trip_validated = callback

    def on_completed(self, callback: Callable):
        """Register callback for completion events."""
        self._on_completed = callback

    def on_error(self, callback: Callable):
        """Register callback for error events."""
        self._on_error = callback

    async def connect(self, session_id: str):
        """
        Connect to the WebSocket and listen for updates.

        Args:
            session_id: The session ID from schedule request
        """
        self.session_id = session_id
        ws_uri = f"{self.ws_url}/validate/live/ws/{session_id}"

        async with websockets.connect(ws_uri) as websocket:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    pass

    async def _handle_message(self, data: dict):
        """Dispatch message to appropriate callback."""
        msg_type = data.get("type")

        if msg_type == "status" and self._on_status:
            await self._call_callback(self._on_status, data)
        elif msg_type == "progress" and self._on_progress:
            await self._call_callback(self._on_progress, data)
        elif msg_type == "trip_validated" and self._on_trip_validated:
            await self._call_callback(self._on_trip_validated, data)
        elif msg_type == "completed" and self._on_completed:
            await self._call_callback(self._on_completed, data)
        elif msg_type == "error" and self._on_error:
            await self._call_callback(self._on_error, data)

    async def _call_callback(self, callback: Callable, data: dict):
        """Call callback, handling both sync and async functions."""
        import asyncio

        result = callback(data)
        if asyncio.iscoroutine(result):
            await result
