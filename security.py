"""Security boundary helpers; does not load models or contact external services."""
import os
import secrets
import time
import warnings
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from flask import abort, request, session, Response
from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 20_000_000

def configure_security(app):
    key = os.environ.get("FLASK_SECRET_KEY", "")
    if len(key) < 32:
        raise RuntimeError("Set FLASK_SECRET_KEY to a random secret of at least 32 characters.")
    app.config.update(SECRET_KEY=key, MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "true").lower() == "true",
        TRUSTED_HOSTS=[x.strip() for x in os.environ.get("TRUSTED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if x.strip()])
    request_limits = {}

    @app.before_request
    def request_boundary():
        # Single-process, per-IP abuse protection; use a shared limiter in deployment.
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} or request.path.startswith("/admin"):
            now = time.monotonic()
            for address, (start, count) in list(request_limits.items()):
                if now - start >= 60:
                    request_limits.pop(address, None)
            address = request.remote_addr or "unknown"
            start, count = request_limits.get(address, (now, 0))
            if count >= 60 or (address not in request_limits and len(request_limits) >= 10000):
                abort(429, description="Too many requests; try again in a minute.")
            request_limits[address] = (start, count + 1)
        # Block legacy static paths too: old files must not become public again.
        if request.path.startswith(("/static/uploads/", "/static/audio/", "/static/admin_uploads/")):
            abort(404)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            source = request.headers.get("Origin") or request.headers.get("Referer")
            if not source:
                abort(403, description="A same-origin browser request is required.")
            src, dst = urlsplit(source), urlsplit(request.host_url)
            if (src.scheme, src.netloc) != (dst.scheme, dst.netloc):
                abort(403, description="Cross-origin request rejected.")

    @app.after_request
    def response_boundary(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

def current_session_id():
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(32)
    return session["sid"]

def is_admin_authenticated():
    user, password = os.environ.get("ADMIN_USERNAME"), os.environ.get("ADMIN_PASSWORD")
    auth = request.authorization
    return bool(user and password and auth and secrets.compare_digest(auth.username or "", user) and secrets.compare_digest(auth.password or "", password))

def admin_required(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        user, password = os.environ.get("ADMIN_USERNAME"), os.environ.get("ADMIN_PASSWORD")
        if not user or not password:
            abort(403, description="Administration is disabled.")
        auth = request.authorization
        if not is_admin_authenticated():
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Admin"'})
        return fn(*args, **kwargs)
    return guarded

def save_validated_upload(upload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid4().hex + ".jpg")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(upload.stream) as img:
                if img.format not in {"JPEG", "PNG"} or img.width * img.height > MAX_PIXELS:
                    abort(400, description="Use a JPEG or PNG of at most 20 million pixels.")
                img.verify()
            upload.stream.seek(0)
            with Image.open(upload.stream) as img:
                # Re-encode pixels: discard EXIF/GPS metadata and supplied filename.
                img.convert("RGB").save(target, "JPEG", quality=90)
        return str(target)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        target.unlink(missing_ok=True)
        abort(400, description="Invalid or unsafe image upload.")
