from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import get_current_username
from src.api.schemas import LoginRequest, MeResponse, TokenResponse
from src.api.security import create_access_token
from src.auth.users import UserRepository

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user_repo = UserRepository.from_env()
    user = user_repo.authenticate(username=payload.username, password=payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.username))


@router.get("/me", response_model=MeResponse)
def me(username: str = Depends(get_current_username)) -> MeResponse:
    return MeResponse(username=username)
