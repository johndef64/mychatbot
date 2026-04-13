"""
Models Manager — browse models by provider, fetch live model lists and pricing.
"""
import streamlit as st
import requests
import sys, os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_api_keys,
    gpt_models, deepseek_models, x_models, groq_models,
    anthropic_models, google_models, alibaba_models, openrouter_models,
    models_by_provider,
)

ss = st.session_state

# ── Theme sync ────────────────────────────────────────────────────────────────
if "dark_mode" not in ss:
    ss["dark_mode"] = True

def apply_theme():
    if ss["dark_mode"]:
        bg="#1e1e2e"; sb="#181825"; tx="#cdd6f4"; sub="#a6adc8"
        inp="#313244"; brd="#45475a"; acc="#89b4fa"; act="#1e1e2e"
        hov="#3d3f55"
    else:
        bg="#f8f9fa"; sb="#e9ecef"; tx="#212529"; sub="#6c757d"
        inp="#ffffff"; brd="#dee2e6"; acc="#0d6efd"; act="#ffffff"
        hov="#e2e6ea"
    st.markdown(f"""<style>
    html,body,[class*="css"],.stApp,.main .block-container{{background:{bg}!important;color:{tx}!important;}}
    section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{{background:{sb}!important;}}
    section[data-testid="stSidebar"] *{{color:{tx}!important;}}
    label,p,span,div,h1,h2,h3,h4,h5,h6,.stMarkdown,[data-testid="stMarkdownContainer"]{{color:{tx}!important;}}
    small,.caption,[data-testid="stCaptionContainer"]{{color:{sub}!important;}}
    .stTextInput>div>div>input,.stTextArea>div>div>textarea{{background:{inp}!important;color:{tx}!important;border-color:{brd}!important;}}
    div[data-baseweb="select"]>div{{background:{inp}!important;color:{tx}!important;border-color:{brd}!important;}}
    div[data-baseweb="menu"] li{{background:{inp}!important;color:{tx}!important;}}
    div[data-baseweb="menu"] li:hover{{background:{hov}!important;}}
    .stButton>button{{background:{inp}!important;color:{tx}!important;border:1px solid {brd}!important;border-radius:6px!important;}}
    .stButton>button:hover{{border-color:{acc}!important;color:{acc}!important;}}
    details,[data-testid="stExpander"],[data-testid="stExpander"]>div{{background:{inp}!important;border-color:{brd}!important;}}
    .stTabs [data-baseweb="tab-list"]{{background:{sb}!important;border-radius:8px!important;padding:4px!important;}}
    .stTabs [data-baseweb="tab"]{{background:transparent!important;color:{sub}!important;border-radius:6px!important;}}
    .stTabs [aria-selected="true"]{{background:{acc}!important;color:{act}!important;border-radius:6px!important;}}
    .stTabs [data-baseweb="tab-panel"]{{background:{bg}!important;}}
    hr{{border-color:{brd}!important;}}
    /* Dataframe */
    [data-testid="stDataFrame"]>div,[data-testid="stDataFrame"] iframe{{background:{inp}!important;color:{tx}!important;}}
    ::-webkit-scrollbar{{width:6px;height:6px;}}
    ::-webkit-scrollbar-track{{background:{bg};}}
    ::-webkit-scrollbar-thumb{{background:{brd};border-radius:3px;}}
    ::-webkit-scrollbar-thumb:hover{{background:{acc};}}
    </style>""", unsafe_allow_html=True)
    return acc

accent = apply_theme()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 MyChatbot v2.0")
    _, c2 = st.columns([3, 1])
    with c2:
        if st.button("🌙" if ss["dark_mode"] else "☀️", key="theme_models"):
            ss["dark_mode"] = not ss["dark_mode"]
            st.rerun()
    st.page_link("MyChatbot.py",      label="💬 Chat",   icon="💬")
    st.page_link("pages/Models.py",   label="🤖 Models", icon="🤖")

api_keys = load_api_keys()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🤖 Models Manager")
st.caption("Browse models by provider · fetch live model lists & pricing")
st.divider()

# ── Fetch functions ───────────────────────────────────────────────────────────
# Every fetcher returns (list_of_dicts, error_string_or_None)
# Each dict has at least: id, context_k, input_usd, output_usd, in_app

ALL_APP_MODELS = set(
    gpt_models + deepseek_models + x_models + groq_models +
    anthropic_models + google_models + alibaba_models + openrouter_models
)

# Static pricing fallback ($/1M tokens, updated Apr 2025)
STATIC_PRICING = {
    "gpt-4.1":          (2.00,  8.00,   1047),
    "gpt-4.1-mini":     (0.40,  1.60,   1047),
    "gpt-4.1-nano":     (0.10,  0.40,   1047),
    "gpt-4o":           (2.50,  10.00,  128),
    "gpt-4o-mini":      (0.15,  0.60,   128),
    "o1-mini":          (1.10,  4.40,   128),
    "claude-opus-4-5":          (15.00, 75.00, 200),
    "claude-sonnet-4-5":        (3.00,  15.00, 200),
    "claude-opus-4-0":          (15.00, 75.00, 200),
    "claude-sonnet-4-0":        (3.00,  15.00, 200),
    "claude-3-7-sonnet-latest": (3.00,  15.00, 200),
    "claude-3-5-sonnet-latest": (3.00,  15.00, 200),
    "claude-3-5-haiku-latest":  (0.80,  4.00,  200),
    "gemini-2.5-pro-preview-06-05":    (1.25,  10.00, 1048),
    "gemini-2.5-flash-preview-05-20":  (0.15,  0.60,  1048),
    "gemini-2.0-flash":                (0.10,  0.40,  1048),
    "gemini-2.0-flash-lite":           (0.075, 0.30,  1048),
    "gemini-1.5-pro":                  (1.25,  5.00,  2097),
    "gemini-1.5-flash":                (0.075, 0.30,  1048),
    "deepseek-chat":     (0.27, 1.10, 64),
    "deepseek-reasoner": (0.55, 2.19, 64),
    "grok-4":        (3.00, 15.00, 256),
    "grok-3":        (3.00, 15.00, 131),
    "grok-3-mini":   (0.30, 0.50,  131),
    "grok-2-latest": (2.00, 10.00, 131),
    "llama-3.3-70b-versatile":                       (0.59, 0.79, 128),
    "deepseek-r1-distill-llama-70b":                 (0.75, 0.99, 128),
    "gemma2-9b-it":                                  (0.20, 0.20, 8),
    "moonshotai/kimi-k2-instruct-0905":              (1.00, 3.00, 128),
    "qwen-qwq-32b":                                  (0.29, 0.39, 32),
    "mistral-saba-24b":                              (0.79, 0.79, 32),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.20, 0.20, 128),
    "meta-llama/llama-4-scout-17b-16e-instruct":     (0.11, 0.34, 16),
    "qwen-max":        (1.60, 6.40,  32),
    "qwen-plus":       (0.40, 1.20,  131),
    "qwen-turbo":      (0.05, 0.20,  1000),
    "qwen3-235b-a22b": (0.22, 0.88,  131),
    "qwen3-30b-a3b":   (0.07, 0.28,  131),
    "qwen3-32b":       (0.07, 0.28,  131),
    "qwq-32b":         (0.07, 0.28,  131),
    "qwen-coder-plus":              (0.40, 1.20, 131),
    "qwen2.5-coder-32b-instruct":   (0.07, 0.28, 131),
}

def sp(model_id):
    """Return (inp_str, out_str, ctx_str) from static pricing or '—'."""
    p = STATIC_PRICING.get(model_id)
    if p:
        return f"${p[0]}", f"${p[1]}", f"{p[2]}k"
    return "—", "—", "—"


def build_rows(app_models, fetched_ids=None, fetched_pricing=None):
    """
    Build a list of row dicts for the dataframe.
    app_models    : list of model IDs currently in the app
    fetched_ids   : list of model IDs from the live API (None = not fetched)
    fetched_pricing: dict  model_id -> (inp_usd, out_usd, ctx_k)  (live pricing)
    """
    import pandas as pd
    rows = []
    seen = set()

    # First: app models
    for m in app_models:
        seen.add(m)
        if fetched_pricing and m in fetched_pricing:
            fp = fetched_pricing[m]
            inp, out, ctx = f"${fp[0]}", f"${fp[1]}", f"{fp[2]}k"
            src = "🌐 live"
        else:
            inp, out, ctx = sp(m)
            src = "📋 static" if inp != "—" else "—"
        rows.append({"Model": m, "Context": ctx,
                     "Input $/1M": inp, "Output $/1M": out,
                     "Pricing": src, "Status": "✅ in app"})

    # Then: models from live fetch not yet in app
    if fetched_ids:
        for m in sorted(fetched_ids):
            if m in seen:
                continue
            seen.add(m)
            if fetched_pricing and m in fetched_pricing:
                fp = fetched_pricing[m]
                inp, out, ctx = f"${fp[0]}", f"${fp[1]}", f"{fp[2]}k"
                src = "🌐 live"
            else:
                inp, out, ctx = sp(m)
                src = "📋 static" if inp != "—" else "—"
            rows.append({"Model": m, "Context": ctx,
                         "Input $/1M": inp, "Output $/1M": out,
                         "Pricing": src, "Status": "➕ new"})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Model","Context","Input $/1M","Output $/1M","Pricing","Status"])


def render_table(df):
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Model":        st.column_config.TextColumn(width="large"),
            "Context":      st.column_config.TextColumn(width="small"),
            "Input $/1M":   st.column_config.TextColumn(width="small"),
            "Output $/1M":  st.column_config.TextColumn(width="small"),
            "Pricing":      st.column_config.TextColumn(width="small"),
            "Status":       st.column_config.TextColumn(width="small"),
        }
    )

# ── Live fetchers (all store into ss to survive reruns) ───────────────────────

def _fetch_openai(key):
    r = requests.get("https://api.openai.com/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json()["data"] if "gpt" in m["id"] or m["id"].startswith("o"))
    return ids, {}

def _fetch_anthropic(key):
    r = requests.get("https://api.anthropic.com/v1/models",
                     headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=10)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    return sorted(ids), {}

def _fetch_google(key):
    r = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=10)
    r.raise_for_status()
    models_raw = r.json().get("models", [])
    ids = []
    pricing = {}
    for m in models_raw:
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        mid = m["name"].replace("models/", "")
        ids.append(mid)
        # Google embeds inputTokenPricePer1MTokens in some responses
        inp = m.get("inputTokenPricePer1MTokens")
        out = m.get("outputTokenPricePer1MTokens")
        ctx = m.get("inputTokenLimit", 0)
        if inp and out:
            pricing[mid] = (float(inp), float(out), ctx // 1000)
    return sorted(ids), pricing

def _fetch_groq(key):
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    ids = sorted(m["id"] for m in data)
    # Groq doesn't expose pricing in the models endpoint
    return ids, {}

def _fetch_deepseek(key):
    r = requests.get("https://api.deepseek.com/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []))
    return ids, {}

def _fetch_xai(key):
    r = requests.get("https://api.x.ai/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []))
    return ids, {}

def _fetch_alibaba(key):
    r = requests.get(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []))
    return ids, {}

def _fetch_openrouter():
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = []
    pricing = {}
    for m in data:
        mid = m.get("id", "")
        if not mid:
            continue
        ids.append(mid)
        try:
            inp = float(m.get("pricing", {}).get("prompt", 0) or 0) * 1_000_000
            out = float(m.get("pricing", {}).get("completion", 0) or 0) * 1_000_000
            ctx = m.get("context_length", 0)
            pricing[mid] = (round(inp, 6), round(out, 6), ctx // 1000)
        except Exception:
            pass
    return sorted(ids), pricing


def do_fetch(provider_key, fetcher_fn, *args):
    """Run a fetch, store result in ss, handle errors."""
    with st.spinner(f"Fetching..."):
        try:
            ids, live_pricing = fetcher_fn(*args)
            ss[f"fetched_{provider_key}_ids"]     = ids
            ss[f"fetched_{provider_key}_pricing"]  = live_pricing
            ss[f"fetched_{provider_key}_ts"]       = time.strftime("%H:%M:%S")
            ss[f"fetched_{provider_key}_err"]      = None
            return True
        except Exception as e:
            ss[f"fetched_{provider_key}_err"] = str(e)
            return False


def fetch_header(provider_key, label):
    """Render the fetch button row and status, return (ids_or_None, pricing_or_None)."""
    col1, col2, col3 = st.columns([4, 1, 2])
    with col2:
        clicked = st.button("🔄 Fetch", key=f"btn_{provider_key}")
    with col3:
        ts = ss.get(f"fetched_{provider_key}_ts")
        if ts:
            st.caption(f"fetched at {ts}")

    if clicked:
        key_needed = {
            "openai":    api_keys.get("openai",""),
            "anthropic": api_keys.get("anthropic",""),
            "google":    api_keys.get("googleai", api_keys.get("gemini","")),
            "groq":      api_keys.get("groq",""),
            "deepseek":  api_keys.get("deepseek",""),
            "xai":       api_keys.get("grok",""),
            "alibaba":   api_keys.get("alibaba",""),
            "openrouter": "public",
        }
        fn_map = {
            "openai":    (_fetch_openai,    [key_needed["openai"]]),
            "anthropic": (_fetch_anthropic, [key_needed["anthropic"]]),
            "google":    (_fetch_google,    [key_needed["google"]]),
            "groq":      (_fetch_groq,      [key_needed["groq"]]),
            "deepseek":  (_fetch_deepseek,  [key_needed["deepseek"]]),
            "xai":       (_fetch_xai,       [key_needed["xai"]]),
            "alibaba":   (_fetch_alibaba,   [key_needed["alibaba"]]),
            "openrouter":(_fetch_openrouter,[]),
        }
        k = key_needed.get(provider_key, "")
        if provider_key != "openrouter" and (not k or k in ("missing","miss","")):
            ss[f"fetched_{provider_key}_err"] = f"No API key for {label}"
        else:
            fn, args = fn_map[provider_key]
            ok = do_fetch(provider_key, fn, *args)

    err = ss.get(f"fetched_{provider_key}_err")
    if err:
        st.error(f"❌ {err}")

    ids     = ss.get(f"fetched_{provider_key}_ids")
    pricing = ss.get(f"fetched_{provider_key}_pricing", {})
    if ids is not None:
        new_count = len([x for x in ids if x not in ALL_APP_MODELS])
        st.success(f"✅ {len(ids)} models — {new_count} not yet in app")
    return ids, pricing


# ── Provider tabs ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🟢 OpenAI", "🟣 Anthropic", "🔴 Google",
    "🟠 Alibaba", "🔵 DeepSeek", "🟡 xAI",
    "⚡ Groq", "🌐 OpenRouter"
])

# ─── OpenAI ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("OpenAI")
    ids, lp = fetch_header("openai", "OpenAI")
    df = build_rows(gpt_models, ids, lp)
    render_table(df)
    st.caption("💡 Pricing live via OpenAI API (input/output per 1M tokens). [Docs](https://platform.openai.com/docs/models)")

# ─── Anthropic ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Anthropic")
    ids, lp = fetch_header("anthropic", "Anthropic")
    df = build_rows(anthropic_models, ids, lp)
    render_table(df)
    st.caption("💡 Anthropic doesn't expose pricing in the API — prices from static table (Apr 2025).")

# ─── Google ───────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Google Gemini")
    ids, lp = fetch_header("google", "Google")
    df = build_rows(google_models, ids, lp)
    render_table(df)
    if lp:
        st.success("🌐 Pricing fetched live from Google AI API")
    else:
        st.caption("💡 Google may include pricing in the models response. Static fallback shown.")
    st.info("ℹ️ Uses `googleai` key (same key from Google AI Studio)")

# ─── Alibaba ──────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Alibaba DashScope (Qwen)")
    ids, lp = fetch_header("alibaba", "Alibaba")
    df = build_rows(alibaba_models, ids, lp)
    render_table(df)
    st.caption("💡 DashScope doesn't include pricing in the models endpoint — prices from static table (Apr 2025).")

# ─── DeepSeek ─────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("DeepSeek")
    ids, lp = fetch_header("deepseek", "DeepSeek")
    df = build_rows(deepseek_models, ids, lp)
    render_table(df)
    st.caption("💡 Pricing from static table (Apr 2025). [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)")

# ─── xAI ──────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("xAI Grok")
    ids, lp = fetch_header("xai", "xAI")
    df = build_rows(x_models, ids, lp)
    render_table(df)
    st.caption("💡 Pricing from static table (Apr 2025). [xAI pricing](https://x.ai/api)")

# ─── Groq ─────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Groq")
    ids, lp = fetch_header("groq", "Groq")
    df = build_rows(groq_models, ids, lp)
    render_table(df)
    st.caption("💡 Groq pricing from static table (Apr 2025). [Groq pricing](https://groq.com/pricing)")

# ─── OpenRouter ───────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("OpenRouter (300+ models)")
    st.caption("Public API — no key needed for model list. Pricing fetched live.")

    ids, lp = fetch_header("openrouter", "OpenRouter")

    if ids is not None:
        import pandas as pd

        # Build full OpenRouter dataframe with live pricing
        rows = []
        for m in ids:
            p = lp.get(m, (None, None, None))
            inp = f"${p[0]:.4f}" if p[0] is not None else "—"
            out = f"${p[1]:.4f}" if p[1] is not None else "—"
            ctx = f"{p[2]}k"     if p[2] is not None else "—"
            rows.append({
                "Model": m,
                "Context": ctx,
                "Input $/1M": inp,
                "Output $/1M": out,
                "Status": "✅ in app" if m in ALL_APP_MODELS else "",
            })
        full_df = pd.DataFrame(rows)

        # Filters
        fc1, fc2, fc3 = st.columns([3, 1, 1])
        with fc1:
            search = st.text_input("🔍 Filter", placeholder="llama, mistral, qwen, free…",
                                   key="or_search", label_visibility="collapsed")
        with fc2:
            free_only = st.checkbox("Free only", value=False)
        with fc3:
            app_only = st.checkbox("In app only", value=False)

        view = full_df.copy()
        if search:
            view = view[view["Model"].str.contains(search, case=False, na=False)]
        if free_only:
            view = view[view["Input $/1M"].isin(["$0.0", "$0.0000", "—", "$0"])]
        if app_only:
            view = view[view["Status"] == "✅ in app"]

        st.caption(f"Showing {len(view)} / {len(full_df)} models")
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={
                         "Model":        st.column_config.TextColumn(width="large"),
                         "Context":      st.column_config.TextColumn(width="small"),
                         "Input $/1M":   st.column_config.TextColumn(width="small"),
                         "Output $/1M":  st.column_config.TextColumn(width="small"),
                         "Status":       st.column_config.TextColumn(width="small"),
                     })
    else:
        # Show current app models before first fetch
        df = build_rows(openrouter_models, None, {})
        render_table(df)
        st.info("👆 Click **Fetch** to load all 300+ OpenRouter models with live pricing (no API key needed).")

st.divider()
st.caption("🔄 Fetch buttons query the provider's live API. Results are cached for this session. "
           "🌐 = pricing from live API · 📋 = static table (Apr 2025) · ➕ = model not yet in app")
