"""Local accounts with revocable server sessions and per-request actor context."""
from __future__ import annotations

import hashlib
import secrets
import time
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
import psycopg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

actor = ContextVar('bom_actor', default=None)
request_id = ContextVar('bom_request_id', default=None)
ph = PasswordHasher()
SESSION_SECONDS = 30 * 24 * 3600


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify_password_hash(encoded: str, password: str) -> bool:
    """Accept the previous server's PBKDF2 hash once, then upgrade on login."""
    if encoded.startswith('pbkdf2_sha256$'):
        try:
            algorithm, rounds_text, salt_text, expected_text = encoded.split('$')
            rounds = int(rounds_text)
            salt, expected = bytes.fromhex(salt_text), bytes.fromhex(expected_text)
            if algorithm != 'pbkdf2_sha256' or not 600_000 <= rounds <= 2_000_000 or len(salt) != 32 or len(expected) != 32:
                return False
            actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, rounds)
            return secrets.compare_digest(actual, expected)
        except (TypeError, ValueError, UnicodeError):
            return False
    try:
        return bool(ph.verify(encoded, password))
    except (VerificationError, InvalidHashError, TypeError, ValueError):
        return False


class Login(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8)


class NewUser(BaseModel):
    username: str = Field(min_length=2, max_length=64, pattern=r'^[a-zA-Z0-9_.-]+$')
    password: str = Field(min_length=8)
    role: str = 'engineer'


class ResetPassword(BaseModel):
    password: str = Field(min_length=8)


class AuthService:
    def __init__(self, repo):
        self.repo = repo
        self.attempts = {}

    def current(self, request):
        token = request.cookies.get('bom_session')
        if not token: raise HTTPException(401,'请登录')
        with self.repo.connection() as conn:
            user = conn.execute("SELECT u.id,u.username,u.role,u.active,u.must_change_password,s.csrf_hash FROM user_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=%s AND s.revoked_at IS NULL AND s.expires_at>now()",(digest(token),)).fetchone()
        if not user or not user['active']: raise HTTPException(401,'登录已失效')
        return user

    def require(self, request, *, allow_change=False):
        user=self.current(request)
        if user['must_change_password'] and not allow_change: raise HTTPException(403,'首次登录必须修改密码')
        return user

    def csrf(self, request, user):
        sent=request.headers.get('x-bom-csrf','')
        if not sent or not secrets.compare_digest(digest(sent),user['csrf_hash']): raise HTTPException(403,'CSRF 校验失败')

    def admin(self, request):
        user=self.require(request)
        if user['role']!='admin': raise HTTPException(403,'需要管理员权限')
        return user


def create_auth_router(repo, *, secure_cookies=False):
    router=APIRouter(prefix='/api/auth')
    service=AuthService(repo)

    @router.post('/login')
    def login(payload: Login, request: Request):
        key=(request.client.host if request.client else 'local',payload.username.lower())
        now=time.monotonic()
        attempts=[t for t in service.attempts.get(key,[]) if now-t<300]
        if len(attempts)>=8: raise HTTPException(429,'登录尝试过多，请稍后再试')
        with repo.connection() as conn:
            user=conn.execute('SELECT * FROM users WHERE username=%s',(payload.username,)).fetchone()
            valid=bool(user and user['active'] and verify_password_hash(user['password_hash'],payload.password))
            if not valid:
                attempts.append(now);service.attempts[key]=attempts
                raise HTTPException(401,'账号或密码错误')
            service.attempts.pop(key,None)
            if user['password_hash'].startswith('pbkdf2_sha256$'):
                conn.execute('UPDATE users SET password_hash=%s,updated_at=now() WHERE id=%s',(ph.hash(payload.password),user['id']))
            token=secrets.token_urlsafe(48);csrf=secrets.token_urlsafe(32)
            conn.execute("SELECT set_config('app.user_id',%s,true)",(str(user['id']),))
            conn.execute('INSERT INTO user_sessions(id,user_id,token_hash,csrf_hash,expires_at) VALUES(%s,%s,%s,%s,%s)',(uuid4(),user['id'],digest(token),digest(csrf),datetime.now(timezone.utc)+timedelta(seconds=SESSION_SECONDS)))
            conn.execute('INSERT INTO audit_events(actor_id,action,request_id,summary) VALUES(%s,%s,%s,%s)',(user['id'],'login',request_id.get(),'登录'))
        from fastapi.responses import JSONResponse
        response=JSONResponse({'id':str(user['id']),'username':user['username'],'role':user['role'],'must_change_password':user['must_change_password'],'csrf':csrf})
        response.set_cookie('bom_session',token,max_age=SESSION_SECONDS,httponly=True,samesite='lax',secure=secure_cookies,path='/')
        response.set_cookie('bom_csrf',csrf,max_age=SESSION_SECONDS,httponly=False,samesite='lax',secure=secure_cookies,path='/')
        return response

    @router.get('/me')
    def me(request: Request):
        u=service.current(request)
        return {'id':str(u['id']),'username':u['username'],'role':u['role'],'must_change_password':u['must_change_password']}

    @router.post('/password')
    def change_password(payload: PasswordChange, request: Request):
        user=service.require(request,allow_change=True);service.csrf(request,user)
        if payload.new_password==payload.old_password: raise HTTPException(400,'新密码不能与当前密码相同')
        with repo.connection(user['id']) as conn:
            row=conn.execute('SELECT password_hash FROM users WHERE id=%s',(user['id'],)).fetchone()
            if not verify_password_hash(row['password_hash'],payload.old_password): raise HTTPException(400,'原密码错误')
            conn.execute('UPDATE users SET password_hash=%s,must_change_password=false,updated_at=now() WHERE id=%s',(ph.hash(payload.new_password),user['id']))
            conn.execute('UPDATE user_sessions SET revoked_at=now() WHERE user_id=%s',(user['id'],))
            conn.execute('INSERT INTO audit_events(actor_id,action,request_id,summary) VALUES(%s,%s,%s,%s)',(user['id'],'password_change',request_id.get(),'修改密码'))
        from fastapi.responses import JSONResponse
        response=JSONResponse({'status':'changed'});response.delete_cookie('bom_session');response.delete_cookie('bom_csrf');return response

    @router.post('/logout')
    def logout(request: Request):
        user=service.require(request,allow_change=True);service.csrf(request,user)
        with repo.connection(user['id']) as conn:
            conn.execute('UPDATE user_sessions SET revoked_at=now() WHERE token_hash=%s',(digest(request.cookies['bom_session']),))
        from fastapi.responses import JSONResponse
        response=JSONResponse({'status':'ok'});response.delete_cookie('bom_session');response.delete_cookie('bom_csrf');return response

    @router.get('/users')
    def users(request: Request):
        service.admin(request)
        with repo.connection() as conn:
            rows=conn.execute('SELECT id,username,role,active,must_change_password,created_at FROM users ORDER BY created_at').fetchall()
        return [{**r,'id':str(r['id'])} for r in rows]

    @router.post('/users')
    def create_user(payload: NewUser,request: Request):
        admin=service.admin(request);service.csrf(request,admin)
        if payload.role not in ('admin','engineer'): raise HTTPException(422,'角色无效')
        try:
            with repo.connection(admin['id']) as conn:
                user_id=uuid4()
                conn.execute('INSERT INTO users(id,username,password_hash,role) VALUES(%s,%s,%s,%s)',(user_id,payload.username,ph.hash(payload.password),payload.role))
                conn.execute('INSERT INTO audit_events(actor_id,action,request_id,summary) VALUES(%s,%s,%s,%s)',(admin['id'],'user_create',request_id.get(),f'创建账号 {payload.username}'))
        except psycopg.errors.UniqueViolation as exc:
            raise HTTPException(409,'账号已存在') from exc
        return {'id':str(user_id),'username':payload.username}

    @router.post('/users/{user_id}/reset')
    def reset_user(user_id: str,payload: ResetPassword,request: Request):
        admin=service.admin(request);service.csrf(request,admin)
        with repo.connection(admin['id']) as conn:
            row=conn.execute('UPDATE users SET password_hash=%s,must_change_password=true,updated_at=now() WHERE id=%s RETURNING username',(ph.hash(payload.password),user_id)).fetchone()
            if not row: raise HTTPException(404,'账号不存在')
            conn.execute('UPDATE user_sessions SET revoked_at=now() WHERE user_id=%s',(user_id,))
            conn.execute('INSERT INTO audit_events(actor_id,action,request_id,summary) VALUES(%s,%s,%s,%s)',(admin['id'],'user_reset',request_id.get(),f'重置账号 {row["username"]} 密码'))
        return {'status':'reset'}

    @router.post('/users/{user_id}/active/{enabled}')
    def set_active(user_id: str,enabled: bool,request: Request):
        admin=service.admin(request);service.csrf(request,admin)
        if str(admin['id'])==user_id and not enabled: raise HTTPException(400,'不能停用当前管理员')
        with repo.connection(admin['id']) as conn:
            row=conn.execute('UPDATE users SET active=%s,updated_at=now() WHERE id=%s RETURNING username',(enabled,user_id)).fetchone()
            if not row: raise HTTPException(404,'账号不存在')
            if not enabled: conn.execute('UPDATE user_sessions SET revoked_at=now() WHERE user_id=%s',(user_id,))
            conn.execute('INSERT INTO audit_events(actor_id,action,request_id,summary) VALUES(%s,%s,%s,%s)',(admin['id'],'user_active',request_id.get(),f'{"启用" if enabled else "停用"}账号 {row["username"]}'))
        return {'status':'ok'}

    return router,service
