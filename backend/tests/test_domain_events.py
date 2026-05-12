"""
Tests for DomainEventBus - the internal domain event system.
"""

from application.domain.internal_events import (
    DiaryFinishedEvent,
    DomainEventBus,
    InternalEvent,
    domain_events,
)


class TestDomainEventBus:
    """Test DomainEventBus subscribe/emit/unsubscribe functionality."""

    def test_subscribe_and_emit(self):
        """Handler should be called when event is emitted."""
        bus = DomainEventBus()
        received = []

        def handler(data):
            """Handler."""
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0] == {"key": "value"}

    def test_unsubscribe(self):
        """Unsubscribed handler should not be called."""
        bus = DomainEventBus()
        received = []

        def handler(data):
            """Handler."""
            received.append(data)

        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        bus.emit("test_event")

        assert len(received) == 0

    def test_unsubscribe_safe(self):
        """Unsubscribing multiple times or when not subscribed should not raise."""
        bus = DomainEventBus()

        def handler(data):
            """Handler."""
            pass

        # Not subscribed
        bus.unsubscribe("test_event", handler)
        
        # Subscribed then unsubscribed twice
        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)

    def test_handler_exception_safe(self):
        """Exception in handler should be caught and logged, not crash the emitter."""
        bus = DomainEventBus()
        results = []

        def bad_handler(data):
            """Bad handler."""
            raise ValueError("Boom")

        def good_handler(data):
            """Good handler."""
            results.append("ok")

        bus.subscribe("test_event", bad_handler)
        bus.subscribe("test_event", good_handler)
        
        # Should not raise
        bus.emit("test_event")
        assert "ok" in results

    def test_typed_event_emit(self):
        """Typed internal events should emit their payload dict."""
        bus = DomainEventBus()
        received = []

        bus.subscribe(DiaryFinishedEvent, lambda data: received.append(data))
        bus.emit(
            DiaryFinishedEvent(
                diary="diary",
                route_id="route",
                observatory="observatory",
                vehicle_type_name="bus",
            )
        )

        assert issubclass(DiaryFinishedEvent, InternalEvent)
        assert received == [
            {
                "diary": "diary",
                "route_id": "route",
                "observatory": "observatory",
                "vehicle_type_name": "bus",
            }
        ]


class TestGlobalDomainEvents:
    """Test that the global domain_events singleton works."""

    def test_global_bus_exists(self):
        """Global domain_events should be a DomainEventBus instance."""
        assert isinstance(domain_events, DomainEventBus)
