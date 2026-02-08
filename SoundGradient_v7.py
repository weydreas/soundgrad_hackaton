# SoundGradient_v7_Full.py

# ---------------------------
# BOOTSTRAP (must be first; stdlib only)
# ---------------------------
import sys
import os
import time
import traceback
from datetime import datetime

APP_VERSION = "v7 (full: UI+token TTL+cache+rate limit+retry)"

LOG_FILE = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "SoundGradient_error.log")


def _log(s: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except Exception:
        pass


def _pause(msg: str = "") -> None:
    if msg:
        print(msg)
        _log(msg)

    # 1) Console pause (python.exe)
    try:
        input("\nНажмите Enter для выхода...")
        return
    except Exception:
        pass

    # 2) No console (pythonw.exe): show MessageBox
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            "Приложение завершилось. Подробности смотри в лог-файле:\n\n" + LOG_FILE,
            "SoundGradient",
            0
        )
        return
    except Exception:
        pass


def _fatal(prefix: str = "APP CRASHED") -> None:
    print(f"\n=== {prefix} ===")
    _log(f"\n=== {prefix} @ {datetime.now().isoformat()} ===")
    tb = traceback.format_exc()
    print(tb)
    _log(tb)
    _pause(f"(Лог продублирован в {LOG_FILE})")


def _excepthook(exc_type, exc, tb):
    print("\n=== UNCAUGHT EXCEPTION ===")
    _log(f"\n=== UNCAUGHT EXCEPTION @ {datetime.now().isoformat()} ===")
    traceback.print_exception(exc_type, exc, tb)
    _log("".join(traceback.format_exception(exc_type, exc, tb)))
    _pause(f"(Лог продублирован в {LOG_FILE})")


sys.excepthook = _excepthook

# ---------------------------
# Third-party imports (guarded)
# ---------------------------
try:
    import base64
    import json
    import random
    import re
    import uuid
    import hashlib
    from dataclasses import dataclass
    from datetime import timezone
    from typing import List, Set, Tuple, Optional, Dict, Any

    import requests
    from dotenv import load_dotenv
    from flask import Flask, request, render_template_string, redirect, url_for, session
    from flask_session import Session

    import warnings
    from urllib3.exceptions import InsecureRequestWarning

    from io import BytesIO
    from PIL import Image
except Exception:
    _fatal("IMPORT ERROR (вероятно, не установлены зависимости)")
    # Не выходим — держим процесс живым, чтобы окно не закрывалось.
    while True:
        time.sleep(1)

load_dotenv()

# ---------------------------
# Config (GigaChat)
# ---------------------------
GIGACHAT_OAUTH_URL = os.getenv("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
GIGACHAT_API_BASE = os.getenv("GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api/v1").rstrip("/")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

TAGS_FILE = os.getenv("TAGS_FILE", "Tags.txt")
SONGS_FILE = os.getenv("SONGS_FILE", "Songs.txt")

SEND_IMAGE_TO_MODEL = os.getenv("SEND_IMAGE_TO_MODEL", "1").strip() not in ("0", "false", "False")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Token/UI/caching controls
TOKEN_TTL_SECONDS = 30 * 60            # 30 minutes (requested)
ANALYZE_COOLDOWN_SECONDS = 5           # local anti-spam
CACHE_TTL_SECONDS = 15 * 60            # cache last results for 15 minutes
CACHE_MAX_ITEMS = 256                  # memory cap

# Global in-memory cache: key -> {"ts": int, "tags": [...], "playlist": [...]}
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# Models
# ---------------------------
@dataclass(frozen=True)
class Song:
    title: str
    tags: Set[str]

# ---------------------------
# Parsing (matches your syntax)
# ---------------------------
SONG_LINE_RE = re.compile(r"^(?P<title>.*?)\s*(?:\((?P<tags>[^)]*)\))?\s*$")


def load_allowed_tags(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл с тегами: {path}")
    tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            tags.append(t)
    if not tags:
        raise ValueError(f"Файл {path} пуст или не содержит тегов.")
    return tags


def parse_songs(path: str, allowed_tags_set: Set[str]) -> List[Song]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл с песнями: {path}")

    songs: List[Song] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = SONG_LINE_RE.match(line)
            if not m:
                continue

            title = (m.group("title") or "").strip()
            tags_blob = (m.group("tags") or "").strip()

            song_tags: Set[str] = set()
            if tags_blob:
                for t in [x.strip() for x in tags_blob.split(",")]:
                    if t and t in allowed_tags_set:
                        song_tags.add(t)

            if title:
                songs.append(Song(title=title, tags=song_tags))

    if not songs:
        raise ValueError(f"Файл {path} не дал ни одной песни после парсинга.")
    return songs

# ---------------------------
# Image compression for prompt (to avoid 413)
# ---------------------------
def compress_image_for_prompt(image_bytes: bytes, max_side: int = 768, target_bytes: int = 220_000) -> bytes:
    """
    Уменьшаем картинку по большей стороне до max_side и сжимаем в JPEG,
    чтобы итоговый размер был примерно <= target_bytes.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    q = 85
    best = None

    while q >= 35:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        best = data
        if len(data) <= target_bytes:
            return data
        q -= 10

    return best if best is not None else image_bytes

# ---------------------------
# SSL verify helper (session-driven)
# ---------------------------
def ssl_verify_value() -> bool:
    """
    True  -> verify SSL certs (default, safe)
    False -> disable verification (unsafe, for debugging / corp MITM)
    """
    disable = bool(session.get("disable_ssl_verify", False))
    v = not disable
    if not v:
        warnings.simplefilter("ignore", InsecureRequestWarning)
    return v

# ---------------------------
# Token helpers
# ---------------------------
def now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _normalize_expires_at(expires_at: int) -> int:
    """
    Some APIs return ms epoch. Normalize to seconds.
    """
    try:
        ea = int(expires_at)
    except Exception:
        return 0
    if ea > 2_000_000_000_000:  # > ~2033 in ms
        return ea // 1000
    return ea


def token_info() -> Dict[str, Any]:
    """
    Returns:
      status: "missing" | "valid" | "expired"
      obtained_at: int|None (unix seconds)
      expires_at: int|None (unix seconds)
      expires_in: int|None (seconds)
      obtained_str/expires_str: str
    """
    token = session.get("gigachat_access_token")
    obtained_at = session.get("gigachat_access_token_obtained_at")
    expires_at = session.get("gigachat_access_token_expires_at")

    if not token:
        return {
            "status": "missing",
            "obtained_at": None,
            "expires_at": None,
            "expires_in": None,
            "obtained_str": "",
            "expires_str": "",
        }

    now = now_ts()
    ea = _normalize_expires_at(expires_at) if expires_at else 0

    # If expires_at not present, fall back to obtained_at + TTL (requested 30 min)
    oa = int(obtained_at) if obtained_at else now
    if ea <= 0:
        ea = oa + TOKEN_TTL_SECONDS

    expires_in = ea - now

    def fmt(ts: int) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    status = "valid" if expires_in > 0 else "expired"
    return {
        "status": status,
        "obtained_at": oa,
        "expires_at": ea,
        "expires_in": expires_in,
        "obtained_str": fmt(oa),
        "expires_str": fmt(ea),
    }


def get_saved_access_token() -> Optional[str]:
    info = token_info()
    if info["status"] != "valid":
        return None
    return str(session.get("gigachat_access_token"))

# ---------------------------
# GigaChat auth
# ---------------------------
def exchange_auth_key_for_token(auth_key_basic: str, scope: str) -> Tuple[str, int]:
    if not auth_key_basic.strip():
        raise RuntimeError("Пустой Authorization key.")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {auth_key_basic.strip()}",
    }
    data = {"scope": scope}

    r = requests.post(
        GIGACHAT_OAUTH_URL,
        headers=headers,
        data=data,
        timeout=30,
        verify=ssl_verify_value(),
    )
    r.raise_for_status()
    obj = r.json()

    access_token = obj.get("access_token")
    expires_at = obj.get("expires_at")
    if not access_token:
        raise RuntimeError(f"Неожиданный ответ OAuth: {obj}")

    # expires_at may be absent; still store 0 and rely on obtained_at+TTL
    return str(access_token), int(expires_at) if expires_at else 0

# ---------------------------
# GigaChat tagging call (with retry/backoff for 429)
# ---------------------------
JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _post_with_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    """
    Retries on 429 with exponential backoff (and uses Retry-After if provided).
    """
    max_attempts = 5
    for attempt in range(max_attempts):
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
            verify=ssl_verify_value(),
        )

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = 2 ** attempt
            else:
                sleep_s = min(30.0, 2 ** attempt)

            time.sleep(sleep_s)
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError("GigaChat rate limit (429): слишком много запросов. Попробуй позже.")


def gigachat_tags_for_image(access_token: str, image_bytes: bytes, allowed_tags: List[str]) -> List[str]:
    if not access_token.strip():
        raise RuntimeError("Нет access token для GigaChat.")

    allowed_tags_str = ", ".join(allowed_tags)

    url = f"{GIGACHAT_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token.strip()}",
    }

    prompt = (
        "Верни 3–8 тегов настроения для изображения.\n"
        "ВАЖНО: можно использовать ТОЛЬКО теги из списка ниже.\n"
        f"Список тегов: [{allowed_tags_str}]\n\n"
        "Формат ответа: строго JSON вида {\"tags\": [\"tag1\", \"tag2\", ...]} без лишнего текста.\n"
        "Без дублей.\n"
    )

    if SEND_IMAGE_TO_MODEL:
        image_bytes = compress_image_for_prompt(image_bytes, max_side=768, target_bytes=220_000)
        b64 = base64.b64encode(image_bytes).decode("ascii")

        if len(b64) > 350_000:
            raise RuntimeError(
                f"Изображение слишком большое даже после сжатия (base64 length={len(b64)}). "
                "Уменьши картинку или снизь max_side/target_bytes."
            )

        prompt += f"\nИзображение (base64 JPEG):\n{b64}\n"
    else:
        prompt += "\nИзображение не передано (режим отладки). Подбери разумные теги."

    payload = {
        "model": GIGACHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    data = _post_with_retry(url, headers, payload, timeout=90)

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Неожиданный формат ответа GigaChat: {data}")

    m = JSON_EXTRACT_RE.search((content or "").strip())
    if not m:
        raise RuntimeError(f"Модель вернула не-JSON: {str(content)[:300]}...")

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise RuntimeError(f"Не удалось распарсить JSON: {m.group(0)[:300]}...")

    tags = obj.get("tags", [])
    if not isinstance(tags, list):
        raise RuntimeError(f"Неверный формат tags: {obj}")

    allowed_set = set(allowed_tags)
    clean: List[str] = []
    for t in tags:
        if isinstance(t, str) and t in allowed_set and t not in clean:
            clean.append(t)

    if not clean:
        clean = random.sample(allowed_tags, k=min(3, len(allowed_tags)))
    return clean

# ---------------------------
# Playlist selection
# ---------------------------
def build_playlist(image_tags: List[str], songs: List[Song], k: int = 3) -> List[Song]:
    itags = set(image_tags)

    scored: List[Tuple[int, Song]] = [(len(s.tags & itags), s) for s in songs]
    positives = [s for score, s in scored if score > 0]
    zeroes = [s for score, s in scored if score == 0]

    random.shuffle(positives)
    random.shuffle(zeroes)
    positives.sort(key=lambda s: len(s.tags & itags), reverse=True)

    picked: List[Song] = []
    for s in positives:
        if len(picked) >= k:
            break
        picked.append(s)

    if len(picked) < k:
        for s in zeroes:
            if len(picked) >= k:
                break
            picked.append(s)

    random.shuffle(picked)
    return picked

# ---------------------------
# Cache helpers
# ---------------------------
def _cache_prune() -> None:
    if not IMAGE_CACHE:
        return
    now = now_ts()

    expired = [k for k, v in IMAGE_CACHE.items() if (now - int(v.get("ts", 0))) > CACHE_TTL_SECONDS]
    for k in expired:
        IMAGE_CACHE.pop(k, None)

    if len(IMAGE_CACHE) > CACHE_MAX_ITEMS:
        items = sorted(IMAGE_CACHE.items(), key=lambda kv: int(kv[1].get("ts", 0)))
        for k, _ in items[: max(0, len(IMAGE_CACHE) - CACHE_MAX_ITEMS)]:
            IMAGE_CACHE.pop(k, None)


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    _cache_prune()
    v = IMAGE_CACHE.get(key)
    if not v:
        return None
    if now_ts() - int(v.get("ts", 0)) > CACHE_TTL_SECONDS:
        IMAGE_CACHE.pop(key, None)
        return None
    return v


def cache_set(key: str, tags: List[str], playlist_titles: List[str]) -> None:
    _cache_prune()
    IMAGE_CACHE[key] = {"ts": now_ts(), "tags": tags, "playlist": playlist_titles}

# ---------------------------
# Web app
# ---------------------------
HTML = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>SoundGradient</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; max-width: 900px; }
    .card { padding: 18px; border: 1px solid #ddd; border-radius: 12px; margin-top: 18px; }
    .row { display: grid; gap: 10px; }
    .err { color: #b00020; white-space: pre-wrap; }
    .ok { color: #0b6; }
    .warn { color: #b26a00; }
    .hint { color: #555; font-size: 14px; }
    .tags span { display: inline-block; padding: 6px 10px; margin: 6px 6px 0 0; border: 1px solid #ccc; border-radius: 999px; }
    input[type=file], input[type=password], select { padding: 10px; border-radius: 10px; border: 1px solid #bbb; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #999; background: #f5f5f5; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    .progress-wrap { margin-top: 10px; height: 8px; background: #eee; border-radius: 999px; overflow: hidden; }
    .progress-bar { width: 0%; height: 8px; background: #999; transition: width 0.2s linear; }

    .loading { display:none; align-items:center; gap:10px; }
    .spinner {
      width: 14px; height: 14px; border: 2px solid #bbb; border-top-color: #444;
      border-radius: 50%; animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>

  <script>
    function setLoading(isLoading){
      const btn = document.getElementById("analyzeBtn");
      const load = document.getElementById("loading");
      const bar = document.getElementById("progressBar");
      if(!btn || !load || !bar) return;

      if(isLoading){
        btn.disabled = true;
        load.style.display = "flex";
        bar.style.width = "10%";

        let w = 10;
        window.__sgTimer = setInterval(()=>{
          w = Math.min(95, w + (w < 70 ? 7 : 2));
          bar.style.width = w + "%";
        }, 300);
      } else {
        btn.disabled = false;
        load.style.display = "none";
        bar.style.width = "0%";
        if(window.__sgTimer){ clearInterval(window.__sgTimer); window.__sgTimer = null; }
      }
    }

    function onAnalyzeSubmit(){
      // hard client-side anti-double-click
      if(window.__sgSubmitting) return false;
      window.__sgSubmitting = true;
      setLoading(true);
      return true;
    }

    window.addEventListener("pageshow", ()=>{ window.__sgSubmitting = false; setLoading(false); });
  </script>
</head>
<body>
  <h1>Mood Playlist (GigaChat) — {{ app_version }}</h1>

  <div class="card">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
      <div>
        <div><b>Token status:</b>
          {% if token_status == 'missing' %}
            <span class="warn">нет токена</span>
          {% elif token_status == 'expired' %}
            <span class="warn">токен устарел</span>
          {% else %}
            <span class="ok">валиден</span>
          {% endif %}
        </div>
        {% if token_status != 'missing' %}
          <div class="hint">
            Получен: {{ token_obtained_str }} · Истекает: {{ token_expires_str }}
            {% if token_status == 'valid' %}
              · Осталось: {{ token_expires_in }} сек
            {% endif %}
          </div>
        {% endif %}
      </div>

      <form action="/set_ssl" method="post" style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        <label class="hint">
          <input type="checkbox" name="disable_ssl" value="1" {% if disable_ssl %}checked{% endif %} />
          Disable SSL verification (unsafe, for debug)
        </label>
        <button type="submit">Apply</button>
        <span class="hint">verify: <b>{% if disable_ssl %}OFF{% else %}ON{% endif %}</b></span>
      </form>
    </div>
  </div>

  <div class="card">
    <form action="/set_auth_key" method="post" class="row">
      <div><b>GigaChat Authorization key</b> <span class="hint">(Basic key: base64(client_id:client_secret))</span></div>
      <input type="password" name="auth_key" placeholder="Введите Authorization key" autocomplete="off" required />
      <div>
        <label class="hint">Scope:</label>
        <select name="scope">
          <option value="GIGACHAT_API_PERS" {% if scope=='GIGACHAT_API_PERS' %}selected{% endif %}>GIGACHAT_API_PERS</option>
          <option value="GIGACHAT_API_B2B" {% if scope=='GIGACHAT_API_B2B' %}selected{% endif %}>GIGACHAT_API_B2B</option>
          <option value="GIGACHAT_API_CORP" {% if scope=='GIGACHAT_API_CORP' %}selected{% endif %}>GIGACHAT_API_CORP</option>
        </select>
      </div>
      <div style="display:flex; gap:10px; align-items:center; flex-wrap: wrap;">
        <button type="submit">Получить токен и сохранить</button>
        {% if token_status == 'valid' %}
          <span class="ok">Токен сохранён и валиден.</span>
          <a class="hint" href="/clear_key">Сбросить</a>
        {% elif token_status == 'expired' %}
          <span class="warn">Токен сохранён, но уже устарел — запроси заново.</span>
          <a class="hint" href="/clear_key">Сбросить</a>
        {% else %}
          <span class="hint">Без токена анализ картинки недоступен, но UI должен открываться.</span>
        {% endif %}
      </div>
      <div class="hint">OAuth: {{ oauth_url }} · API: {{ api_base }} · Model: {{ model }}</div>
      <div class="hint">Если консоль не видна, смотри лог: {{ log_file }}</div>
    </form>
  </div>

  <div class="card">
    <form id="analyzeForm" action="/analyze" method="post" enctype="multipart/form-data" class="row" onsubmit="return onAnalyzeSubmit();">
      <div><b>Картинка</b></div>
      <input type="file" name="image" accept="image/*" required />
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        <button id="analyzeBtn" type="submit" {% if token_status != 'valid' %}disabled{% endif %}>Собрать плейлист</button>
        <div id="loading" class="loading">
          <div class="spinner"></div>
          <div class="hint">Анализирую…</div>
        </div>
      </div>
      <div class="progress-wrap"><div id="progressBar" class="progress-bar"></div></div>
      <div class="hint">
        Песни не проигрываются — выводятся только названия.
        {% if cooldown_left and cooldown_left > 0 %}
          · Локальный cooldown: подожди {{ cooldown_left }} сек
        {% endif %}
      </div>
    </form>
  </div>

  {% if error %}
    <div class="card err">{{ error }}</div>
  {% endif %}

  {% if tags %}
    <div class="card">
      <h3>Теги настроения</h3>
      <div class="tags">
        {% for t in tags %}<span>{{ t }}</span>{% endfor %}
      </div>
      {% if cache_hit %}
        <div class="hint">Источник: cache</div>
      {% endif %}
    </div>
  {% endif %}

  {% if playlist %}
    <div class="card">
      <h3>Плейлист (3 трека, случайный порядок)</h3>
      <ol>
        {% for s in playlist %}
          <li>{{ s }}</li>
        {% endfor %}
      </ol>
    </div>
  {% endif %}
</body>
</html>
"""

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)


def render_page(**kwargs):
    tinfo = token_info()
    last_ts = int(session.get("last_analyze_ts", 0) or 0)
    left = max(0, ANALYZE_COOLDOWN_SECONDS - (now_ts() - last_ts))
    base = dict(
        app_version=APP_VERSION,
        disable_ssl=bool(session.get("disable_ssl_verify", False)),
        scope=session.get("gigachat_scope", "GIGACHAT_API_PERS"),
        oauth_url=GIGACHAT_OAUTH_URL,
        api_base=GIGACHAT_API_BASE,
        model=GIGACHAT_MODEL,
        log_file=LOG_FILE,
        token_status=tinfo["status"],
        token_obtained_str=tinfo.get("obtained_str", ""),
        token_expires_str=tinfo.get("expires_str", ""),
        token_expires_in=tinfo.get("expires_in", ""),
        cooldown_left=left,
        tags=None,
        playlist=None,
        cache_hit=False,
        error=None,
    )
    base.update(kwargs)
    return render_template_string(HTML, **base)


@app.get("/")
def index():
    return render_page()


@app.post("/set_ssl")
def set_ssl():
    disable = request.form.get("disable_ssl") == "1"
    session["disable_ssl_verify"] = disable
    return redirect(url_for("index"))


@app.post("/set_auth_key")
def set_auth_key():
    try:
        auth_key = (request.form.get("auth_key") or "").strip()
        scope = (request.form.get("scope") or "GIGACHAT_API_PERS").strip()
        if not auth_key:
            raise RuntimeError("Введите Authorization key.")

        access_token, expires_at = exchange_auth_key_for_token(auth_key, scope)

        session["gigachat_access_token"] = access_token
        session["gigachat_access_token_expires_at"] = int(expires_at) if expires_at else 0
        session["gigachat_access_token_obtained_at"] = now_ts()
        session["gigachat_scope"] = scope

        return redirect(url_for("index"))
    except Exception as e:
        _log(f"[set_auth_key error] {e}\n{traceback.format_exc()}")
        return render_page(error=str(e)), 400


@app.get("/clear_key")
def clear_key():
    session.pop("gigachat_access_token", None)
    session.pop("gigachat_access_token_expires_at", None)
    session.pop("gigachat_access_token_obtained_at", None)
    return redirect(url_for("index"))


@app.post("/analyze")
def analyze():
    # server-side anti-double-submit + cooldown
    in_progress = bool(session.get("analyze_in_progress", False))
    if in_progress:
        return render_page(error="Анализ уже выполняется. Подожди завершения."), 409

    last_ts = int(session.get("last_analyze_ts", 0) or 0)
    now = now_ts()
    if now - last_ts < ANALYZE_COOLDOWN_SECONDS:
        left = ANALYZE_COOLDOWN_SECONDS - (now - last_ts)
        return render_page(error=f"Слишком быстро. Подожди {left} сек."), 429

    token = get_saved_access_token()
    if not token:
        return render_page(error="Токен отсутствует или устарел. Запроси новый токен."), 401

    session["analyze_in_progress"] = True
    session["last_analyze_ts"] = now

    try:
        if "image" not in request.files:
            raise RuntimeError("Файл не получен.")
        f = request.files["image"]
        if not f.filename:
            raise RuntimeError("Пустое имя файла.")
        image_bytes = f.read()
        if not image_bytes:
            raise RuntimeError("Пустой файл.")

        allowed_tags = load_allowed_tags(TAGS_FILE)
        songs = parse_songs(SONGS_FILE, set(allowed_tags))

        key = hashlib.sha256(image_bytes).hexdigest()
        cached = cache_get(key)
        if cached:
            return render_page(tags=cached["tags"], playlist=cached["playlist"], cache_hit=True), 200

        tags = gigachat_tags_for_image(token, image_bytes, allowed_tags)
        playlist_songs = build_playlist(tags, songs, k=3)
        playlist_titles = [s.title for s in playlist_songs]

        cache_set(key, tags, playlist_titles)

        return render_page(tags=tags, playlist=playlist_titles, cache_hit=False), 200

    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        _log(f"[HTTPError] {e}\n{traceback.format_exc()}")
        if status == 429:
            return render_page(error="GigaChat: 429 Too Many Requests. Подожди и повтори."), 429
        return render_page(error=f"Ошибка GigaChat API: {str(e)}"), 502
    except Exception as e:
        _log(f"[analyze error] {e}\n{traceback.format_exc()}")
        return render_page(error=str(e)), 500
    finally:
        session["analyze_in_progress"] = False


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)
    except Exception:
        _fatal("APP CRASHED (runtime)")
        while True:
            time.sleep(1)
    finally:
        _pause(f"(Если окно не видно — смотри лог: {LOG_FILE})")
