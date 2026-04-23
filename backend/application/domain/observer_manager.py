"""
ObserverManager - Manages observers and their diaries.
"""

from application.domain.live_data import Autobus
from .observers import Observer, Diary
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.domain.virtual_entities import Observatory


class ObserverManager:
    """Manages the lifecycle of observers and provides diary search/retrieval."""

    def __init__(self):
        """Initialize with an empty vehicle-id → Observer map."""
        self.observers = {}

    def create_observer(
        self,
        vehicle: Autobus,
        observatory: "Observatory",
        scheduled_start_time: str = None,
    ) -> Observer:
        """Create an observer for a vehicle and register it."""
        observer = Observer(observatory, vehicle, None)
        observer.start_new_trip(vehicle.trip, scheduled_start_time)
        vehicle.set_observer(observer)
        self.observers[vehicle.id] = observer
        return observer

    def get_observer(self, vehicle_id: str) -> Observer:
        """Get observer by vehicle ID."""
        return self.observers.get(vehicle_id)

    def get_all_observers(self) -> dict:
        """Get all registered observers."""
        return self.observers

    def remove_observer(self, vehicle_id: str):
        """Remove an observer by vehicle ID."""
        if vehicle_id in self.observers:
            del self.observers[vehicle_id]

    def search_diary(self, trip_id: str) -> Diary:
        """Search for an active diary by trip ID."""
        for observer in self.observers.values():
            if observer.current_diary and observer.current_diary.trip_id == trip_id:
                return observer.current_diary
        return None

    def search_history(self, trip_id: str) -> Diary:
        """Search for a diary in history by trip ID."""
        for observer in self.observers.values():
            for diary in observer.diary_history:
                if diary.trip_id == trip_id:
                    return diary
        return None

    def get_completed_diaries(self) -> list:
        """Returns a list of dictionaries from all completed diaries."""
        completed = []
        for observer in self.observers.values():
            # Collect finished diaries from history
            for diary in observer.diary_history:
                completed.extend(diary.to_dict_list())

            # Include current active diary if it has data (graceful exit)
            if observer.current_diary and observer.current_diary.measurements:
                completed.extend(observer.current_diary.to_dict_list())

        return completed

    def get_all_current_diaries(self) -> tuple:
        """Returns (diaries_list, diary_count, observer_count)."""
        all_diaries = []
        for observer in self.observers.values():
            if observer.current_diary and observer.current_diary.measurements:
                all_diaries.extend(observer.current_diary.to_dict_list())
        return all_diaries, len(all_diaries), len(self.observers)

    def get_id_by_label(self, label: str) -> str | None:
        """Find vehicle ID by label."""
        for obs in self.observers.values():
            if obs.assignedVehicle.label == label:
                return obs.assignedVehicle.id
        return None

    def get_vehicle_diaries(self, vehicle_id: str) -> list:
        """Returns a list of dictionaries from all diaries for a specific vehicle."""
        observer = self.observers.get(vehicle_id)
        diaries = []
        if observer:
            # Add history
            if observer.diary_history:
                diaries.extend(observer.diary_history)
            # Add current
            if observer.current_diary:
                diaries.append(observer.current_diary)
        return diaries
