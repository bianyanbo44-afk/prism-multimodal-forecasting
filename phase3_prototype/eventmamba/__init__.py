"""Minimal event-driven multimodal state-space prototype."""

from .data import AsyncEvent, AsyncSample, SyntheticAsyncDataset
from .baselines import DecayEventGRUBaseline, EventGRUBaseline
from .model import EventMamba, EventMambaConfig
from .selective import SelectiveEventMamba

__all__ = [
    "AsyncEvent",
    "AsyncSample",
    "SyntheticAsyncDataset",
    "EventMamba",
    "EventMambaConfig",
    "SelectiveEventMamba",
    "EventGRUBaseline",
    "DecayEventGRUBaseline",
]
