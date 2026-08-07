import bcrypt
from datetime import datetime, timedelta, timezone
import jwt

from app.core.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    ADMIN_USERNAME, ADMIN_PASSWORD_HASH,
    VIEWER_USERNAME, VIEWER_PASSWORD_HASH,
)


def _create_token(subject: str, role: str, expires_delta: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_access_token(subject: str, role: str) -> str:
    return _create_token(subject, role, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")


def create_refresh_token(subject: str, role: str) -> str:
    return _create_token(subject, role, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh")


def decode_token(token: str) -> dict:
    """Raises jwt.ExpiredSignatureError if expired, jwt.InvalidTokenError
    if malformed/bad signature. Callers must catch both."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage. Run once per credential,
    then paste the result into .env. Never store plaintext passwords."""
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a login attempt's password against the stored hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )

def authenticate_user(username: str, password: str) -> str | None:
    """Returns the matched role ('admin' / 'viewer') or None if no match."""
    if username == ADMIN_USERNAME and ADMIN_PASSWORD_HASH:
        if verify_password(password, ADMIN_PASSWORD_HASH):
            return "admin"
    if username == VIEWER_USERNAME and VIEWER_PASSWORD_HASH:
        if verify_password(password, VIEWER_PASSWORD_HASH):
            return "viewer"
    return None