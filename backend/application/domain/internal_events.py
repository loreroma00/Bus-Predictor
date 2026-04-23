"""
Domain Events - Pub/Sub for internal domain logic.

Implements the EventBus interface.
"""

import logging
from typing import Callable, Dict, List, Optional
from .interfaces import EventBus

# Event Constants
DIARY_FINISHED = "diary_finished"


class DomainEventBus:
    """
    Event bus for domain-initiated events (e.g. Diary Completed).
    """

    def __init__(self):
        """Initialize with an empty subscriber map."""
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
                logging.error(f"Error in domain event handler for '{event_name}': {e}")


# Global instance for domain events
domain_events = DomainEventBus()
