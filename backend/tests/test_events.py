"""
Tests for ConsoleEventBus - the event system for console-initiated events.
"""

from interaction.events import ConsoleEventBus, console_events


class TestConsoleEventBus:
    """Test ConsoleEventBus subscribe/emit/unsubscribe functionality."""

    def test_subscribe_and_emit(self):
        """Handler should be called when event is emitted."""
        bus = ConsoleEventBus()
        received = []

        def handler(data):
            """Handler."""
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0] == {"key": "value"}

    def test_emit_without_data(self):
        """Emit should work without data argument."""
        bus = ConsoleEventBus()
        received = []

        def handler(data):
            """Handler."""
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.emit("test_event")

        assert len(received) == 1
        assert received[0] == {}

    def test_multiple_subscribers(self):
        """Multiple handlers should all be called."""
        bus = ConsoleEventBus()
        results = []

        def handler1(data):
            """Handler1."""
            results.append("handler1")

        def handler2(data):
            """Handler2."""
            results.append("handler2")

        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        bus.emit("test_event")

        assert len(results) == 2
        assert "handler1" in results
        assert "handler2" in results

    def test_unsubscribe(self):
        """Unsubscribed handler should not be called."""
        bus = ConsoleEventBus()
        received = []

        def handler(data):
            """Handler."""
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        bus.emit("test_event")

        assert len(received) == 0

    def test_emit_unknown_event(self):
        """Emitting unknown event should not raise."""
        bus = ConsoleEventBus()
        # Should not raise
        bus.emit("nonexistent_event", {"data": "test"})

    def test_handler_exception_does_not_stop_others(self):
        """Exception in one handler should not prevent others from running."""
        bus = ConsoleEventBus()
        results = []

        def bad_handler(data):
            """Bad handler."""
            raise ValueError("Intentional error")

        def good_handler(data):
            """Good handler."""
            results.append("success")

        bus.subscribe("test_event", bad_handler)
        bus.subscribe("test_event", good_handler)
        bus.emit("test_event")

        assert "success" in results

    def test_different_events_are_independent(self):
        """Handlers for different events should not interfere."""
        bus = ConsoleEventBus()
        results = []

        def handler_a(data):
            """Handler a."""
            results.append("A")

        def handler_b(data):
            """Handler b."""
            results.append("B")

        bus.subscribe("event_a", handler_a)
        bus.subscribe("event_b", handler_b)

        bus.emit("event_a")

        assert results == ["A"]

    def test_unsubscribe_not_subscribed(self):
        """Unsubscribing a handler that was never subscribed should not raise."""
        bus = ConsoleEventBus()

        def handler(data):
            """Handler."""
            pass

        # Should not raise
        bus.unsubscribe("test_event", handler)

    def test_unsubscribe_multiple_times(self):
        """Unsubscribing the same handler multiple times should not raise."""
        bus = ConsoleEventBus()

        def handler(data):
            """Handler."""
            pass

        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        # Second call should not raise
        bus.unsubscribe("test_event", handler)

    def test_unsubscribe_unknown_event(self):
        """Unsubscribing from an event that has no subscribers should not raise."""
        bus = ConsoleEventBus()

        def handler(data):
            """Handler."""
            pass

        # Should not raise
        bus.unsubscribe("unknown_event", handler)


class TestGlobalConsoleEvents:
    """Test that the global console_events singleton works."""

    def test_global_bus_exists(self):
        """Global console_events should be a ConsoleEventBus instance."""
        assert isinstance(console_events, ConsoleEventBus)

    def test_global_bus_is_same_instance(self):
        """Importing console_events multiple times should return same instance."""
        from interaction.events import console_events as bus1
        from interaction.events import console_events as bus2

        assert bus1 is bus2
