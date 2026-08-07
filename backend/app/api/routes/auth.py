import jwt
from fastapi import APIRouter, HTTPException, status

from app.core.security import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    role = authenticate_user(payload.username, payload.password)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return TokenResponse(
        access_token=create_access_token(payload.username, role),
        refresh_token=create_refresh_token(payload.username, role),
    )


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh(payload: RefreshRequest):
    try:
        claims = decode_token(payload.refresh_token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

    if claims.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is not a refresh token")

    return AccessTokenResponse(
        access_token=create_access_token(claims["sub"], claims["role"])
    )