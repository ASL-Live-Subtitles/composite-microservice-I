from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from auth.jwt import decode_jwt

security = HTTPBearer()

def require_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    try:
        return decode_jwt(creds.credentials)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

def require_roles(*roles: str):
    def checker(user: Dict[str, Any] = Depends(require_user)):
        if not set(user.get("roles", [])).intersection(roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user
    return checker
