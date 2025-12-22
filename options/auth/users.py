from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class User:
    username: str
    password_hash: str

    @classmethod
    def from_plaintext(cls, username: str, password: str) -> "User":
        return cls(username=username, password_hash=hash_password(password))


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


class UserRepository:
    def __init__(self, users: Dict[str, User]):
        self.users = users

    @classmethod
    def from_env(cls) -> "UserRepository":
        raw = os.getenv("APP_USERS")
        if raw:
            parsed = json.loads(raw)
            users = {
                username: User(username=username, password_hash=hash_password(pw))
                for username, pw in parsed.items()
            }
            return cls(users)

        # default single demo user
        demo_user = User.from_plaintext("admin", "demo123")
        return cls({demo_user.username: demo_user})

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.users.get(username)
        if user and verify_password(password, user.password_hash):
            return user
        return None
