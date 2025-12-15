import os
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any

JWT_SECRET = os.environ["SESSION_SECRET"]
JWT_ALG = "HS256"

def create_jwt(user: Dict[str, Any], roles: list[str]) -> str:
    payload = {
        "sub": user["sub"],
        "email": user["email"],
        "name": user.get("name"),
        "roles": roles,
        "exp": datetime.utcnow() + timedelta(hours=8),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_jwt(token: str) -> Dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
