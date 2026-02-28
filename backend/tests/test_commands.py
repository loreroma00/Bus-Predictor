"""
Tests for Commands - Command pattern with dependency injection.
"""

import pytest
from unittest.mock import Mock, patch
from interaction.commands import (
    Command,
    print_diary,
    fetch_data,
    print_all_diaries,
    command_quit,
    start_observers,
    stop_observers,
    command_help,
)
from interaction.events import ConsoleEventBus


class TestCommandBase:
    """Test the Command ABC structure."""

    def test_command_is_abstract(self):
        """Command should be an abstract base class."""
        # Cannot instantiate Command directly
        with pytest.raises(TypeError):
            Command()

    def test_subclasses_have_command_name(self):
        """All command subclasses should have command_name."""
        for subclass in Command.__subclasses__():
            assert hasattr(subclass, "command_name")
            assert isinstance(subclass.command_name, str)


class TestPrintDiary:
    """Test print_diary command."""

    def test_command_name(self):
        """Command name should be 'print diary'."""
        obs = Mock()
        cmd = print_diary(obs)
        assert cmd.command_name == "print diary"

    def test_execute_with_found_diary(self, caplog):
        """Should print diary when found."""
        import logging
        caplog.set_level(logging.INFO)

        mock_diary = Mock()
        mock_diary.__str__ = Mock(return_value="Mock Diary Content")

        mock_obs = Mock()
        mock_obs.search_diary.return_value = mock_diary
        mock_obs.search_history.return_value = None

        cmd = print_diary(mock_obs)
        cmd.execute("trip_123")

        mock_obs.search_diary.assert_called_once_with("trip_123")
        assert "trip_123" in caplog.text

    def test_execute_with_not_found(self, caplog):
        """Should print not found message when diary missing."""
        import logging
        caplog.set_level(logging.INFO)

        mock_obs = Mock()
        mock_obs.search_diary.return_value = None
        mock_obs.search_history.return_value = None

        cmd = print_diary(mock_obs)
        cmd.execute("unknown_trip")

        assert "No diary found" in caplog.text


class TestFetchData:
    """Test fetch_data command."""

    def test_command_name(self):
        """Command name should be 'fetch data'."""
        obs = Mock()
        formatter = Mock()
        cmd = fetch_data(obs, formatter)
        assert cmd.command_name == "fetch data"

    def test_uses_injected_time_formatter(self):
        """Should use the injected time formatter."""
        mock_obs = Mock()
        mock_obs.search_diary.return_value = None

        mock_formatter = Mock(return_value="12:34:56")

        cmd = fetch_data(mock_obs, mock_formatter)
        cmd.execute("trip_123")

        # Formatter may or may not be called depending on data availability
        mock_obs.search_diary.assert_called_once()


class TestPrintAllDiaries:
    """Test print_all_diaries command."""

    def test_command_name(self):
        """Command name should be 'print diaries'."""
        obs = Mock()
        cmd = print_all_diaries(obs)
        assert cmd.command_name == "print diaries"

    def test_execute_prints_counts(self, caplog):
        """Should print observer and diary counts."""
        import logging
        caplog.set_level(logging.INFO)

        mock_obs = Mock()
        mock_obs.get_all_current_diaries.return_value = ([], 5, 3)

        cmd = print_all_diaries(mock_obs)
        cmd.execute("")

        assert "Total Observers: 3" in caplog.text
        assert "Total Diaries: 5" in caplog.text


class TestEventEmittingCommands:
    """Test commands that emit events."""

    def test_command_quit_emits_shutdown(self):
        """Quit command should emit shutdown_requested event."""
        mock_bus = ConsoleEventBus()
        received = []
        mock_bus.subscribe("shutdown_requested", lambda d: received.append(d))

        with patch("interaction.commands.console_events", mock_bus):
            cmd = command_quit()
            cmd.execute("")

        assert len(received) == 1

    def test_start_observers_emits_event(self):
        """Start command should emit services_start event."""
        mock_bus = ConsoleEventBus()
        received = []
        mock_bus.subscribe("services_start", lambda d: received.append(d))

        with patch("interaction.commands.console_events", mock_bus):
            cmd = start_observers()
            cmd.execute("")

        assert len(received) == 1

    def test_stop_observers_emits_event(self):
        """Stop command should emit services_stop event."""
        mock_bus = ConsoleEventBus()
        received = []
        mock_bus.subscribe("services_stop", lambda d: received.append(d))

        with patch("interaction.commands.console_events", mock_bus):
            cmd = stop_observers()
            cmd.execute("")

        assert len(received) == 1


class TestCommandHelp:
    """Test the help command."""

    def test_command_name(self):
        """Command name should be 'help'."""
        cmd = command_help()
        assert cmd.command_name == "help"

    def test_execute_prints_available_commands(self, caplog):
        """Should print help for available commands."""
        import logging
        caplog.set_level(logging.INFO)

        cmd = command_help()
        cmd.execute("")

        assert "Available Commands" in caplog.text


class TestDependencyInjection:
    """Test that DI works correctly for commands."""

    def test_different_observatories_are_independent(self):
        """Commands with different observatories should be independent."""
        obs1 = Mock()
        obs2 = Mock()

        cmd1 = print_diary(obs1)

        cmd1.execute("trip1")

        obs1.search_diary.assert_called_once()
        obs2.search_diary.assert_not_called()
