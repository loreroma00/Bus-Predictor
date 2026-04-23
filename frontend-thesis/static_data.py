"""
Static data handling - fetches GTFS data and parses routes/trips.
"""

import os
import zipfile
import hashlib
import requests
from typing import Optional

URL_STATIC = "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip"
URL_MD5 = "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip.md5"

HEADERS = {
    "Referer": "https://romamobilita.it/sistemi-e-tecnologie/open-data/",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


class StaticDataFetcher:
    """Fetches and manages GTFS static data files."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the instance."""
        self.base_path = base_path or os.getcwd()
        self.zip_path = os.path.join(self.base_path, "rome_static_gtfs.zip")

    def fetch(self) -> Optional[str]:
        """
        Ensures static data is available. Downloads if needed.
        Returns the authoritative MD5 hash.
        """
        local_md5 = self.compute_md5() if os.path.exists(self.zip_path) else None

        remote_md5: Optional[str] = None
        try:
            remote_md5 = self._download_if_needed(local_md5)
        except Exception as e:
            print(f"Error downloading static data: {e}. Trying local file...")

        final_md5 = remote_md5 if remote_md5 else local_md5

        if os.path.exists(self.zip_path):
            try:
                self._unzip()
            except Exception as e:
                print(f"Error unzipping: {e}")

        return final_md5

    def compute_md5(self) -> str:
        """Compute MD5 hash of the local zip file."""
        md5 = hashlib.md5()
        with open(self.zip_path, "rb") as f:
            while chunk := f.read(4096):
                md5.update(chunk)
        return md5.hexdigest()

    def _download_if_needed(self, local_md5: Optional[str] = None) -> str:
        """
        Check remote MD5 and download if different.
        Returns the remote MD5.
        """
        response = requests.get(URL_MD5, headers=HEADERS, timeout=15)
        remote_md5 = response.text.strip()

        if local_md5 is None or local_md5.strip() not in remote_md5:
            print(f"Downloading static GTFS data (New Version: {remote_md5})...")
            content = requests.get(URL_STATIC, headers=HEADERS, timeout=60).content
            with open(self.zip_path, "wb") as f:
                f.write(content)

        return remote_md5

    def _unzip(self):
        """Extract the zip file to base directory."""
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.base_path)
