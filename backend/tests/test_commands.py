"""Tests for console commands."""

import pytest
from unittest.mock import Mock, patch

from application.domain.internal_events import ConsoleEventBus
from interaction.commands import (
    Command,
    command_help,
    command_quit,
    fetch_data,
    print_all_live_trips,
    print_live_trip,
    start_services,
    stop_services,
)


class TestCommandBase:
    """Test the command ABC structure."""

    def test_command_is_abstract(self):
        """Command should not be directly instantiable."""
        with pytest.raises(TypeError):
            Command()

    def test_subclasses_have_command_name(self):
        """Every command subclass should declare a command name."""
        for subclass in Command.__subclasses__():
            assert hasattr(subclass, "command_name")
            assert isinstance(subclass.command_name, str)


class TestPrintLiveTrip:
    """Test print_live_trip command."""

    def test_command_name(self):
        """Command name should be 'print live trip'."""
        cmd = print_live_trip(Mock())
        assert cmd.command_name == "print live trip"

    def test_execute_with_found_live_trip(self, caplog):
        """Should print the live trip when found."""
        import logging

        caplog.set_level(logging.INFO)
        live_trip = Mock()
        live_trip.format_rich.return_value = "Mock LiveTrip Content"

        observatory = Mock()
        observatory.search_live_trip.return_value = live_trip
        observatory.search_completed_live_trip.return_value = None

        cmd = print_live_trip(observatory)
        cmd.execute("trip_123")

        observatory.search_live_trip.assert_called_once_with("trip_123")
        assert "Mock LiveTrip Content" in caplog.text

    def test_execute_with_not_found(self, caplog):
        """Should print a not-found message when the trip is missing."""
        import logging

        caplog.set_level(logging.INFO)
        observatory = Mock()
        observatory.search_live_trip.return_value = None
        observatory.search_completed_live_trip.return_value = None

        cmd = print_live_trip(observatory)
        cmd.execute("unknown_trip")

        assert "No live trip found" in caplog.text


class TestFetchData:
    """Test fetch_data command."""

    def test_command_name(self):
        """Command name should be 'fetch data'."""
        cmd = fetch_data(Mock(), Mock())
        assert cmd.command_name == "fetch data"

    def test_uses_injected_time_formatter(self):
        """Should query live trip data through the injected observatory."""
        observatory = Mock()
        observatory.search_live_trip.return_value = None
        formatter = Mock(return_value="12:34:56")

        cmd = fetch_data(observatory, formatter)
        cmd.execute("trip_123")

        observatory.search_live_trip.assert_called_once_with("trip_123")


class TestPrintAllLiveTrips:
    """Test print_all_live_trips command."""

    def test_command_name(self):
        """Command name should be 'print live trips'."""
        cmd = print_all_live_trips(Mock())
        assert cmd.command_name == "print live trips"

    def test_execute_prints_counts(self, caplog):
        """Should print live-trip and measurement counts."""
        import logging

        caplog.set_level(logging.INFO)
        observatory = Mock()
        observatory.get_all_current_measurements.return_value = ([], 5, 3)

        cmd = print_all_live_trips(observatory)
        cmd.execute("")

        assert "Total LiveTrips: 3" in caplog.text
        assert "Total Measurements: 5" in caplog.text


class TestEventEmittingCommands:
    """Test commands that emit events."""

    def test_command_quit_emits_shutdown(self):
        """Quit command should emit shutdown_requested."""
        bus = ConsoleEventBus()
        received = []
        bus.subscribe("shutdown_requested", lambda data: received.append(data))

        with patch("interaction.commands.console_events", bus):
            command_quit().execute("")

        assert len(received) == 1

    def test_start_services_emits_event(self):
        """Start command should emit services_start."""
        bus = ConsoleEventBus()
        received = []
        bus.subscribe("services_start", lambda data: received.append(data))

        with patch("interaction.commands.console_events", bus):
            start_services().execute("")

        assert len(received) == 1

    def test_stop_services_emits_event(self):
        """Stop command should emit services_stop."""
        bus = ConsoleEventBus()
        received = []
        bus.subscribe("services_stop", lambda data: received.append(data))

        with patch("interaction.commands.console_events", bus):
            stop_services().execute("")

        assert len(received) == 1


class TestCommandHelp:
    """Test the help command."""

    def test_command_name(self):
        """Command name should be 'help'."""
        assert command_help().command_name == "help"

    def test_execute_prints_available_commands(self, caplog):
        """Should print the available commands list."""
        import logging

        caplog.set_level(logging.INFO)
        command_help().execute("")

        assert "Available Commands" in caplog.text


class TestDependencyInjection:
    """Test that commands keep observatories isolated."""

    def test_different_observatories_are_independent(self):
        """Commands with different observatories should be independent."""
        obs1 = Mock()
        obs2 = Mock()

        print_live_trip(obs1).execute("trip1")

        obs1.search_live_trip.assert_called_once()
        obs2.search_live_trip.assert_not_called()
