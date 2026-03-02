"""
API client for the ATAC Bus Delay Prediction backend.
"""

import requests
from typing import Optional

API_URL = "https://atacapi.loreromaphotos.it"


class APIClient:
    """HTTP client for the prediction API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or API_URL).rstrip("/")

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
        Request model validation for a specific date.

        Args:
            date: DD-MM-YYYY format

        Returns:
            Validation response dict with error metrics and trip summaries
        """
        url = f"{self.base_url}/validate"
        payload = {"date": date}
        response = requests.post(url, json=payload, timeout=300)  # Validation can take longer
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
