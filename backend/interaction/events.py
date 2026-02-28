"""
ConsoleEventBus - Pub/sub for console-initiated events.

Events: shutdown_requested, services_start, services_stop
"""

import logging
from typing import Callable, Dict, List, Optional
from application.domain.interfaces import EventBus


class ConsoleEventBus(EventBus):
    """
    Event bus for console/user-initiated events.

    Usage:
        console_events.subscribe("shutdown_requested", handler_fn)
        console_events.emit("shutdown_requested")
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, handler: Callable):
        """Subscribe a handler to an event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Callable):
        """Remove a handler from an event."""
        if event_name in self._subscribers and handler in self._subscribers[event_name]:
            self._subscribers[event_name].remove(handler)

    def emit(self, event_name: str, data: Optional[dict] = None):
        """
        Emit an event. All subscribed handlers are called synchronously.
        """
        if event_name not in self._subscribers:
            return

        for handler in self._subscribers[event_name]:
            try:
                handler(data or {})
            except Exception as e:
                logging.error(f"Error in event handler for '{event_name}': {e}")


# Global instance for console events
console_events = ConsoleEventBus()


# ============================================================
# Event Names (for reference)
# ============================================================
# "shutdown_requested"     - User wants to quit
# "services_start"         - Start collection/saving
# "services_stop"          - Stop collection/saving
