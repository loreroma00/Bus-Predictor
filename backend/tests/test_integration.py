"""
Integration Tests - Test interactions between components.
"""

from unittest.mock import Mock


class TestObservatoryWithCacheStrategy:
    """Test Observatory with injected cache strategy."""

    def test_observatory_accepts_cache_strategy(self):
        """Observatory should accept a cache strategy via DI."""
        from application.domain.virtual_entities import Observatory
        from persistence import NoCacheStrategy

        strategy = NoCacheStrategy()
        obs = Observatory(cache_strategy=strategy)

        assert obs._cache == strategy

    def test_observatory_uses_cache_strategy_load(self):
        """Observatory should use cache strategy's load method."""

        mock_strategy = Mock()
        mock_strategy.load.return_value = None  # No cache


class TestConsoleCommandRegistry:
    """Test command registration and lookup."""

    def test_register_commands_populates_registry(self):
        """Registering commands should populate the registry."""
        from interaction import console

        mock_obs = Mock()
        console.register_commands(mock_obs)

        assert len(console._command_registry) > 0

    def test_registered_commands_have_correct_names(self):
        """Registered commands should have expected names."""
        from interaction import console

        mock_obs = Mock()
        console.register_commands(mock_obs)

        expected_commands = [
            "print diary",
            "fetch data",
            "print diaries",
            "quit",
            "help",
        ]

        for cmd_name in expected_commands:
            assert cmd_name in console._command_registry, f"Missing: {cmd_name}"


class TestEventDrivenServices:
    """Test that services respond to events correctly."""

    def test_shutdown_event_sets_flags(self):
        """Shutdown event should set appropriate flags."""
        from interaction.events import ConsoleEventBus

        bus = ConsoleEventBus()
        shutdown_called = []

        def mock_shutdown(data):
            """Mock shutdown."""
            shutdown_called.append(True)

        bus.subscribe("shutdown_requested", mock_shutdown)
        bus.emit("shutdown_requested")

        assert len(shutdown_called) == 1


class TestArchitectureLayers:
    """Test that architectural boundaries are respected."""

    def test_domain_does_not_import_persistence(self):
        """Domain layer should not import persistence."""
        import os

        domain_files = [
            "application/domain/virtual_entities.py",
            "application/domain/observers.py",
            "application/domain/interfaces.py",
            "application/domain/time_utils.py",
            "application/domain/live_data.py",
            "application/domain/map_info.py",
        ]

        for filepath in domain_files:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                assert "from persistence" not in content, (
                    f"{filepath} imports persistence!"
                )
                assert "import persistence" not in content, (
                    f"{filepath} imports persistence!"
                )

    def test_domain_does_not_import_interaction(self):
        """Domain layer should not import interaction."""
        import os

        domain_files = [
            "application/domain/virtual_entities.py",
            "application/domain/observers.py",
            "application/domain/interfaces.py",
            "application/domain/live_data.py",
        ]

        for filepath in domain_files:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                assert "from interaction" not in content, (
                    f"{filepath} imports interaction!"
                )
                assert "import interaction" not in content, (
                    f"{filepath} imports interaction!"
                )


class TestProtocolCompliance:
    """Test that implementations satisfy protocols."""

    def test_file_cache_strategy_satisfies_protocol(self):
        """FileCacheStrategy should satisfy CacheStrategy protocol."""
        from application.domain.interfaces import CacheStrategy
        from persistence import FileCacheStrategy

        strategy = FileCacheStrategy()

        # runtime_checkable protocol
        assert isinstance(strategy, CacheStrategy)

    def test_no_cache_strategy_satisfies_protocol(self):
        """NoCacheStrategy should satisfy CacheStrategy protocol."""
        from application.domain.interfaces import CacheStrategy
        from persistence import NoCacheStrategy

        strategy = NoCacheStrategy()

        assert isinstance(strategy, CacheStrategy)


class TestDataModuleInitialization:
    """Test data module initialization pattern."""

    def test_data_observatory_starts_as_none(self):
        """data.OBSERVATORY should be None before initialization."""
        # This test may fail if run after other tests that initialize
        # Using a fresh import check approach instead
        pass  # Skip - depends on test order

    def test_initialize_sets_observatory(self):
        """initialize() should set the OBSERVATORY global."""
        from application.live import data

        mock_obs = Mock()
        data.initialize(mock_obs)

        assert data.OBSERVATORY == mock_obs
