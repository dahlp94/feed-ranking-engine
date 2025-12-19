from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

class EventType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"

@dataclass(frozen=True)
class User:
    user_id: int

@dataclass(frozen=True)
class Post:
    post_id: int
    author_id: int
    topic_id: int
    created_ts: int  # integer "minutes since start"

@dataclass(frozen=True)
class Event:
    user_id: int
    post_id: int
    ts: int
    event_type: EventType

@dataclass
class Session:
    """
    One ranking instance: user sees a set of impressions, clicks subset.
    """
    user_id: int
    ts: int
    impression_post_ids: List[int]
    clicked_post_ids: List[int]

@dataclass
class SimData:
    users: List[User]
    posts: List[Post]
    follows: Dict[int, List[int]]  # user_id -> list of author_ids followed
    sessions: List[Session]
