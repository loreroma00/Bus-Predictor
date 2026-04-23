"""
Trip Verification Strategies - Pluggable filters for validating trip data quality.
Uses Strategy pattern to allow swapping verification logic (e.g., basic rules vs ML).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.domain.observers import Diary
    from application.domain.static_data import Shape


class TripVerificationStrategy(ABC):
    """Strategy interface for verifying if a trip was properly served."""

    @abstractmethod
    def is_trip_valid(self, diary: "Diary", shape: "Shape" = None) -> bool:
        """
        Returns True if the diary represents a valid, completed trip.

        Args:
            diary: The diary to validate
            shape: Optional shape for spatial validation
        """
        pass


class BasicTripVerification(TripVerificationStrategy):
    """
    Basic verification using measurement count and shape coverage.

    Criteria:
    1. Minimum number of measurements (default: 7)
    2. Optional: minimum shape coverage ratio
    """

    def __init__(
        self,
        min_measurements: int = 7,
        min_shape_coverage: float = 0.0,
    ):
        """
        Args:
            min_measurements: Minimum number of measurements required
            min_shape_coverage: Minimum ratio of shape covered (0.0-1.0)
                               Set to 0.0 to disable shape checking
        """
        self.min_measurements = min_measurements
        self.min_shape_coverage = min_shape_coverage

    def is_trip_valid(self, diary: "Diary", shape: "Shape" = None) -> bool:
        """Return True if the diary meets the minimum-measurement and coverage thresholds."""
        # No measurements = invalid
        if not diary.measurements:
            return False

        # Check minimum measurement count
        if len(diary.measurements) < self.min_measurements:
            return False

        # Shape coverage check (optional)
        if self.min_shape_coverage > 0.0 and shape is not None:
            coverage = self._calculate_shape_coverage(diary, shape)
            if coverage < self.min_shape_coverage:
                return False

        return True

    def _calculate_shape_coverage(self, diary: "Diary", shape: "Shape") -> float:
        """
        Calculate what fraction of the route shape was covered.

        Returns ratio in [0.0, 1.0].
        """
        if not diary.measurements or shape is None:
            return 0.0

        # Get total shape length
        total_length = float(shape.distances[-1]) if len(shape.distances) > 0 else 0.0
        if total_length <= 0:
            return 0.0

        # Project first and last measurements onto shape
        first_m = diary.measurements[0]
        last_m = diary.measurements[-1]

        first_dist = shape.project(first_m.gpsdata.latitude, first_m.gpsdata.longitude)
        last_dist = shape.project(last_m.gpsdata.latitude, last_m.gpsdata.longitude)

        # Coverage = distance traveled / total length
        distance_covered = abs(last_dist - first_dist)
        return min(1.0, distance_covered / total_length)


class ScaledMeasurementVerification(TripVerificationStrategy):
    """
    Verification that scales minimum measurements based on route length.

    Uses stops count as a proxy for expected measurements.
    """

    def __init__(
        self,
        measurements_per_stop: float = 0.5,
        absolute_minimum: int = 7,
        min_shape_coverage: float = 0.3,
    ):
        """
        Args:
            measurements_per_stop: Expected measurements per stop on route
            absolute_minimum: Floor for minimum measurements
            min_shape_coverage: Minimum shape coverage ratio
        """
        self.measurements_per_stop = measurements_per_stop
        self.absolute_minimum = absolute_minimum
        self.min_shape_coverage = min_shape_coverage

    def is_trip_valid(
        self, diary: "Diary", shape: "Shape" = None, stop_count: int = None
    ) -> bool:
        """Return True if the diary meets a dynamic minimum scaled by ``stop_count``."""
        if not diary.measurements:
            return False

        # Calculate dynamic minimum based on stop count
        if stop_count and stop_count > 0:
            dynamic_min = max(
                self.absolute_minimum,
                int(stop_count * self.measurements_per_stop),
            )
        else:
            dynamic_min = self.absolute_minimum

        if len(diary.measurements) < dynamic_min:
            return False

        # Shape coverage check
        if self.min_shape_coverage > 0.0 and shape is not None:
            coverage = BasicTripVerification._calculate_shape_coverage(
                BasicTripVerification(), diary, shape
            )
            if coverage < self.min_shape_coverage:
                return False

        return True
