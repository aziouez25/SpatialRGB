#!/usr/bin/env python3
# ==============================================================================
# auth.py — a login password for the SpatialRGB app.
#
# WHY THIS EXISTS (author, 2026-10-05, before deploying to a server)
#   The app binds 0.0.0.0 and has never had any access control: anything that
#   could reach port 8765 could read it.  That was tolerable only because the
#   one route in was an SSH tunnel.  On a server it is not.
#
# WHY ASGI MIDDLEWARE AND NOT A LOGIN PAGE INSIDE THE APP
#   A Shiny app has three doors, and a reactive "are you logged in?" gate only
#   closes one of them:
#     1. GET /                        the page
#     2. WebSocket /websocket/        every input and every rendered output
#     3. GET /session/<id>/download/  the full-resolution H&E export
#   A login form guards (1) and leaves (2) and (3) open to anyone who knows the
#   URL shape.  Middleware wraps the ASGI app, so all three are behind the same
#   check, and an unauthenticated WebSocket is closed at the handshake.
#
# THE WEBSOCKET WRINKLE
#   A browser cannot set headers on a WebSocket upgrade.  Browsers DO usually
#   replay cached Basic credentials to the same origin, but "usually" is not a
#   design.  So a successful HTTP auth also sets a short-lived signed cookie,
#   and the WebSocket is accepted on EITHER the header or that cookie.  The
#   signing secret is random per process, so restarting the app logs everyone
#   out, which is the behaviour you want after a config change.
#
# WHAT THIS IS NOT
#   Basic auth sends the password base64-encoded, which is not encryption.  It
#   is only meaningful over HTTPS or inside an SSH tunnel.  Deployed on plain
#   HTTP it protects against casual access, NOT against anyone who can watch the
#   network.  Put it behind a TLS-terminating reverse proxy on a real server.
#
# CREDENTIALS, in order of precedence
#   1. $SPATIALRGB_USER / $SPATIALRGB_PASSWORD   (handy for systemd or Docker)
#   2. a hashed credentials file, default ~/.spatialrgb_auth.json, overridable
#      with $SPATIALRGB_AUTH_FILE.  PBKDF2-SHA256, stdlib only.
#   With neither set the app REFUSES TO START rather than serving openly --
#   failing closed, because the failure mode of the alternative is silent.
#
# Usage
#   set a password:   ../spatial-venv/bin/python SpatialRGB/auth.py --set
#       --set prompts, so it needs a REAL TERMINAL.  Commands issued through an
#       agent session have no controlling tty and getpass dies at EOF before it
#       can ask -- hence --set-stdin, which also covers systemd, Docker build
#       args and any other unattended install.
#   non-interactive:  printf 'pw' | ... auth.py --set-stdin --user NAME
#   check one:        ../spatial-venv/bin/python SpatialRGB/auth.py --check
# ==============================================================================

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

ITERATIONS = 240_000
COOKIE = "spatialrgb_auth"
COOKIE_TTL = 12 * 3600
DEFAULT_FILE = os.path.expanduser("~/.spatialrgb_auth.json")
_SECRET = secrets.token_bytes(32)            # per process: a restart logs out


def auth_file():
    return os.environ.get("SPATIALRGB_AUTH_FILE", DEFAULT_FILE)


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return dict(salt=salt.hex(), hash=dk.hex(), iterations=iterations)


def save_password(user, password, path=None):
    path = path or auth_file()
    rec = dict(user=user, **hash_password(password))
    old = os.umask(0o077)                    # the file must not be world-readable
    try:
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    return path


def load_credentials():
    """(user, verify_fn) or (None, None) when nothing is configured."""
    u = os.environ.get("SPATIALRGB_USER")
    p = os.environ.get("SPATIALRGB_PASSWORD")
    if u and p:
        return u, lambda pw: hmac.compare_digest(pw, p)
    path = auth_file()
    if os.path.exists(path):
        rec = json.load(open(path))
        salt = bytes.fromhex(rec["salt"])
        want = bytes.fromhex(rec["hash"])
        it = int(rec.get("iterations", ITERATIONS))

        def verify(pw, salt=salt, want=want, it=it):
            got = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, it)
            return hmac.compare_digest(got, want)      # constant time, always
        return rec["user"], verify
    return None, None


# ------------------------------------------------------------------- cookie
def _token(user, ttl=COOKIE_TTL):
    exp = int(time.time()) + ttl
    msg = f"{user}|{exp}".encode()
    sig = hmac.new(_SECRET, msg, hashlib.sha256).hexdigest()
    return f"{base64.urlsafe_b64encode(msg).decode()}.{sig}"


def _token_ok(tok, user):
    try:
        b64, sig = tok.split(".", 1)
        msg = base64.urlsafe_b64decode(b64.encode())
        if not hmac.compare_digest(
                hmac.new(_SECRET, msg, hashlib.sha256).hexdigest(), sig):
            return False
        who, exp = msg.decode().rsplit("|", 1)
        return who == user and int(exp) > time.time()
    except Exception:                                   # noqa: BLE001
        return False


def _header(scope, name):
    name = name.lower().encode()
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v.decode("latin-1")
    return None


class BasicAuth:
    """ASGI middleware: Basic auth on HTTP, header-or-cookie on WebSocket."""

    def __init__(self, app, user, verify, realm="SpatialRGB"):
        self.app, self.user, self.verify, self.realm = app, user, verify, realm

    def _authorised(self, scope):
        h = _header(scope, "authorization")
        if h and h.startswith("Basic "):
            try:
                raw = base64.b64decode(h[6:]).decode("utf-8", "replace")
                u, _, pw = raw.partition(":")
            except Exception:                           # noqa: BLE001
                return False
            # compare the user in constant time too, so the reply does not
            # take a different amount of time for a wrong username
            if hmac.compare_digest(u, self.user) and self.verify(pw):
                return True
        cookie = _header(scope, "cookie") or ""
        for part in cookie.split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and _token_ok(v, self.user):
                return True
        return False

    async def __call__(self, scope, receive, send):
        t = scope.get("type")
        if t not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not self._authorised(scope):
            if t == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                body = b"SpatialRGB: authentication required.\n"
                await send({"type": "http.response.start", "status": 401,
                            "headers": [
                                (b"www-authenticate",
                                 f'Basic realm="{self.realm}"'.encode()),
                                (b"content-type", b"text/plain; charset=utf-8"),
                                (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
            return
        if t == "http" and _header(scope, "authorization"):
            # refresh the cookie so the WebSocket that follows is accepted even
            # if the browser declines to replay the header on the upgrade
            async def send_wrap(msg):
                if msg["type"] == "http.response.start":
                    msg = dict(msg)
                    msg["headers"] = list(msg.get("headers", [])) + [
                        (b"set-cookie",
                         f"{COOKIE}={_token(self.user)}; Path=/; HttpOnly; "
                         f"SameSite=Lax; Max-Age={COOKIE_TTL}".encode())]
                await send(msg)
            await self.app(scope, receive, send_wrap)
            return
        await self.app(scope, receive, send)


def protect(app):
    """Wrap a Shiny App. Refuses to run unprotected — fails closed."""
    user, verify = load_credentials()
    if not user:
        raise SystemExit(
            "SpatialRGB: no password configured, refusing to start.\n"
            f"  set one:  ../spatial-venv/bin/python SpatialRGB/auth.py --set\n"
            f"  or export SPATIALRGB_USER and SPATIALRGB_PASSWORD\n"
            f"  (credentials file looked for at {auth_file()})")
    return BasicAuth(app, user, verify)


if __name__ == "__main__":
    import getpass
    import sys
    if "--set-stdin" in sys.argv:
        # the password arrives on stdin; nothing is echoed and nothing is stored
        # in shell history if the caller pipes it from read -s or a secret store
        u = "spatial"
        if "--user" in sys.argv:
            u = sys.argv[sys.argv.index("--user") + 1]
        pw = sys.stdin.readline().rstrip("\n")
        if len(pw) < 8:
            raise SystemExit("too short — use at least 8 characters")
        print(f"user {u!r}: written to {save_password(u, pw)} (mode 0600)")
    elif "--set" in sys.argv:
        u = input("username: ").strip() or "spatial"
        p1 = getpass.getpass("password: ")
        if len(p1) < 8:
            raise SystemExit("too short — use at least 8 characters")
        if p1 != getpass.getpass("again: "):
            raise SystemExit("they do not match")
        print(f"written to {save_password(u, p1)} (mode 0600)")
    elif "--check" in sys.argv:
        u, v = load_credentials()
        if not u:
            raise SystemExit(f"no credentials: {auth_file()} absent and "
                             "SPATIALRGB_USER/PASSWORD unset")
        src = ("environment" if os.environ.get("SPATIALRGB_PASSWORD")
               else auth_file())
        print(f"user {u!r}, from {src}")
        print("password correct" if v(getpass.getpass("password: "))
              else "password WRONG")
    else:
        print(__doc__ or "use --set or --check")
