import os, sys, contextlib, re, socket
import pickle
from openai import OpenAI
import streamlit as st
import base64
from utils import *
from assistants import *
import pyperclip as pc
from image_gen import (
    OPENAI_IMAGE_MODELS, GOOGLE_IMAGE_MODELS,
    OPENROUTER_IMAGE_MODELS, ALIBABA_IMAGE_MODELS,
    DEFAULT_IMAGE_MODELS, IMAGE_TRIGGERS,
    is_image_prompt, generate_image, save_image,
)

save_log = True
ss = st.session_state

# ─── Theme ────────────────────────────────────────────────────────────────────
if "dark_mode" not in ss:
    ss["dark_mode"] = False

def apply_theme():
    if ss["dark_mode"]:
        bg          = "#1e1e2e"
        sidebar_bg  = "#181825"
        text        = "#cdd6f4"
        subtext     = "#a6adc8"
        input_bg    = "#313244"
        border      = "#45475a"
        accent      = "#89b4fa"
        accent_txt  = "#1e1e2e"
        msg_bot     = "#252535"
        code_bg     = "#11111b"
        hover_bg    = "#3d3f55"
        warning_bg  = "#3d3020"
        success_bg  = "#1e3328"
        info_bg     = "#1e2a3d"
        upload_bg   = "#252535"
        tab_active  = accent
        tab_txt_act = accent_txt
    else:
        bg          = "#f8f9fa"
        sidebar_bg  = "#e9ecef"
        text        = "#212529"
        subtext     = "#6c757d"
        input_bg    = "#ffffff"
        border      = "#dee2e6"
        accent      = "#0d6efd"
        accent_txt  = "#ffffff"
        msg_bot     = "#f8f9fa"
        code_bg     = "#f1f3f5"
        hover_bg    = "#e2e6ea"
        warning_bg  = "#fff3cd"
        success_bg  = "#d1e7dd"
        info_bg     = "#cfe2ff"
        upload_bg   = "#f0f2f6"
        tab_active  = accent
        tab_txt_act = "#ffffff"

    st.markdown(f"""
    <style>
    /* ── Root & body ── */
    html, body, [class*="css"], .stApp {{
        background-color: {bg} !important;
        color: {text} !important;
    }}

    /* ── Main content area ── */
    .main .block-container {{
        background-color: {bg} !important;
        color: {text} !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background-color: {sidebar_bg} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background-color: {input_bg} !important;
        border-color: {border} !important;
        color: {text} !important;
    }}

    /* ── Labels & text ── */
    label, p, span, div, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, [data-testid="stMarkdownContainer"] {{
        color: {text} !important;
    }}
    small, .caption, [data-testid="stCaptionContainer"] {{
        color: {subtext} !important;
    }}

    /* ── Text inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    input[type="text"], input[type="password"], textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {border} !important;
        caret-color: {text} !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 2px {accent}44 !important;
    }}

    /* ── Selectbox / dropdowns ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {text} !important;
    }}
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] ul {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    div[data-baseweb="menu"] li {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    div[data-baseweb="menu"] li:hover {{
        background-color: {hover_bg} !important;
    }}
    div[data-baseweb="menu"] [aria-selected="true"] {{
        background-color: {accent}33 !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 6px !important;
    }}
    .stButton > button:hover {{
        border-color: {accent} !important;
        color: {accent} !important;
        background-color: {hover_bg} !important;
    }}
    .stDownloadButton > button {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    /* ── Chat messages ── */
    .stChatMessage {{
        background-color: {msg_bot} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }}
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] * {{
        color: {text} !important;
    }}

    /* ── Chat input ── */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea,
    .stChatInputContainer,
    .stChatInputContainer > div,
    .stChatInputContainer textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {accent} !important;
    }}
    [data-testid="stChatInput"] button {{
        background-color: {accent} !important;
        color: {accent_txt} !important;
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] > div,
    section[data-testid="stFileUploadDropzone"] {{
        background-color: {upload_bg} !important;
        border-color: {border} !important;
        color: {text} !important;
    }}
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] small {{
        color: {subtext} !important;
    }}
    [data-testid="stFileUploadDropzone"] button {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}

    /* ── Expander ── */
    details, [data-testid="stExpander"],
    [data-testid="stExpander"] > div {{
        background-color: {input_bg} !important;
        border-color: {border} !important;
        color: {text} !important;
    }}
    details summary {{
        color: {text} !important;
    }}

    /* ── Checkboxes ── */
    .stCheckbox label span {{
        color: {text} !important;
    }}

    /* ── Alerts / info boxes ── */
    [data-testid="stAlert"] {{
        color: {text} !important;
    }}
    .stAlert [data-baseweb="notification"] {{
        background-color: {info_bg} !important;
        color: {text} !important;
    }}
    div[data-testid="stNotification"],
    div.element-container div.stInfo,
    div.element-container div.stSuccess,
    div.element-container div.stWarning,
    div.element-container div.stError {{
        color: {text} !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {sidebar_bg} !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {subtext} !important;
        border-radius: 6px !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {hover_bg} !important;
        color: {text} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_active} !important;
        color: {tab_txt_act} !important;
        border-radius: 6px !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {bg} !important;
    }}

    /* ── Divider ── */
    hr {{ border-color: {border} !important; }}

    /* ── Code ── */
    code, pre, .stCode {{
        background-color: {code_bg} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {{ color: {accent} !important; }}
    [data-testid="stMetricLabel"] {{ color: {subtext} !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {accent}; }}

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] * {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ─── Multi-chat session management ────────────────────────────────────────────
MAX_CHATS = 6

if "num_chats" not in ss:
    ss["num_chats"] = 1
if "active_chat" not in ss:
    ss["active_chat"] = 0

def chat_key(i, key):
    return f"chat{i}_{key}"

def init_chat(i):
    if chat_key(i, "assistant_name") not in ss:
        ss[chat_key(i, "assistant_name")] = "none"
    if chat_key(i, "format_name") not in ss:
        ss[chat_key(i, "format_name")] = "base"
    if chat_key(i, "model_name") not in ss:
        ss[chat_key(i, "model_name")] = "meta-llama/llama-4-scout-17b-16e-instruct"
    if chat_key(i, "messages") not in ss:
        ass_name = ss[chat_key(i, "assistant_name")]
        if "assistant" not in ss:
            ss["assistant"] = assistants[ass_name]
        ss[chat_key(i, "messages")] = [{"role": "system", "content": ss["assistant"]}]
    if chat_key(i, "sys_addings") not in ss:
        ss[chat_key(i, "sys_addings")] = []
    if chat_key(i, "reply") not in ss:
        ss[chat_key(i, "reply")] = ""
    if chat_key(i, "title") not in ss:
        ss[chat_key(i, "title")] = f"Chat {i+1}"

for i in range(MAX_CHATS):
    init_chat(i)

# ─── Assistants ───────────────────────────────────────────────────────────────
format_list = list(features['reply_style'].keys())

assistant_list = [
    'none', 'base', 'creator', 'fixer', 'novelist', 'delamain', 'oracle', 'snake', 'roger',
    'leonardo', 'galileo', 'newton',
    'mendel', 'watson', 'crick', 'venter',
    'collins', 'elsevier', 'springer',
    'darwin', 'dawkins',
    'penrose', 'turing', 'marker',
    'mike', 'michael', 'julia', 'jane', 'yoko', 'asuka', 'misa', 'hero', 'xiao', 'peng',
    'miguel', 'francois', 'luca',
    'english', 'spanish', 'french', 'italian', 'portuguese',
    'korean', 'chinese', 'japanese', 'japanese_teacher', 'portuguese_teacher'
]

try:
    from assistants_ext import extra
except ImportError:
    extra = {}
assistants.update(extra)
assistant_list.extend(extra.keys())

if "assistant" not in ss:
    ss["assistant"] = assistants["none"]

# ─── API Keys ─────────────────────────────────────────────────────────────────
if len(list(load_api_keys().keys())) > 0:
    api_keys = load_api_keys()
    ss.openai_api_key    = api_keys.get("openai",    "missing")
    ss.gemini_api_key    = api_keys.get("gemini",    "missing")
    ss.googleai_api_key  = api_keys.get("googleai",  "missing")
    ss.deepseek_api_key  = api_keys.get("deepseek",  "missing")
    ss.x_api_key         = api_keys.get("grok",      "missing")
    ss.groq_api_key      = api_keys.get("groq",      "missing")
    ss.anthropic_api_key = api_keys.get("anthropic", "missing")
    ss.alibaba_api_key   = api_keys.get("alibaba",   "missing")
    ss.openrouter_api_key= api_keys.get("openrouter","missing")
else:
    ss.openai_api_key = ss.gemini_api_key = ss.googleai_api_key = None
    ss.deepseek_api_key = ss.x_api_key = ss.groq_api_key = None
    ss.anthropic_api_key = ss.alibaba_api_key = ss.openrouter_api_key = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

if 'app_initialized' not in ss:
    local_ip = get_local_ip()
    print("\n" + "#"*62)
    print("##      MYCHATBOT PRO - MULTI-AI ASSISTANT v2.0           ##")
    print("#"*62)
    print(f"|| STREAMING: http://{local_ip}:8501")
    print("|| MODELS: OpenAI | DeepSeek | Grok | Groq | Anthropic | Google | Alibaba")
    print("-"*62 + "\n")
    ss['app_initialized'] = True

def encode_ioimage(uploaded_image):
    image_data = uploaded_image.read()
    return base64.b64encode(image_data).decode("utf-8")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def strip_think_tag(input_string):
    pattern = r"<think>(.*?)</think>"
    think_part = re.findall(pattern, input_string, re.DOTALL)
    senza_think = re.sub(pattern, '', input_string, flags=re.DOTALL).strip()
    think_part = think_part[0].strip() if think_part else ''
    return senza_think, think_part

def remove_system_entries(input_list):
    return [e for e in input_list if e.get('role') != 'system']

def update_assistant_in_chat(chat_msgs, assistant_content, sys_addings):
    updated = remove_system_entries(chat_msgs)
    updated.append({"role": "system", "content": assistant_content})
    for add in sys_addings:
        updated.append({"role": "system", "content": add})
    return updated

def remove_last_non_system(input_list):
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i].get('role') != 'system':
            del input_list[i]
            break
    return input_list

def generate_chat_name(path, assistant_name):
    index = 1
    while os.path.exists(os.path.join(path, f"{assistant_name}_{index}.pkl")):
        index += 1
    return f"{assistant_name}_{index}"

def save_chat_as_pickle(chat_idx, path='chats/'):
    if not os.path.exists(path):
        os.mkdir(path)
    ass_name = ss[chat_key(chat_idx, "assistant_name")]
    chat_name = generate_chat_name(path, ass_name)
    with open(os.path.join(path, chat_name + '.pkl'), 'wb') as file:
        pickle.dump(ss[chat_key(chat_idx, "messages")], file)
    return chat_name

def load_chat_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def export_chat_as_markdown(chat_idx):
    msgs = ss[chat_key(chat_idx, "messages")]
    ass_name = ss[chat_key(chat_idx, "assistant_name")]
    model_id = ss[chat_key(chat_idx, "model_name")]
    if len(msgs) <= 1:
        return "# Empty Chat\n\nNo messages to export."
    md = f"# Chat Export - {ass_name}\n\n"
    md += f"**Model:** {model_id}\n**Assistant:** {ass_name}\n"
    md += f"**Export Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in msgs:
        if msg['role'] == 'user':
            md += f"## User\n\n{msg['content']}\n\n"
        elif msg['role'] == 'assistant':
            md += f"## {ass_name}\n\n{msg['content']}\n\n"
        elif msg['role'] == 'system':
            md += f"*System: {msg['content']}*\n\n"
    return md

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 MyChatbot v2.0")

    # Theme toggle
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.caption("Appearance")
    with col_t2:
        theme_label = "🌙" if ss["dark_mode"] else "☀️"
        if st.button(theme_label, key="theme_toggle", help="Toggle light/dark theme"):
            ss["dark_mode"] = not ss["dark_mode"]
            st.rerun()

    st.divider()

    # API Keys
    with st.expander("🔑 API Keys", expanded=not any([
            ss.openai_api_key, ss.deepseek_api_key, ss.x_api_key,
            ss.groq_api_key, ss.anthropic_api_key])):
        providers_keys = [
            ("OpenAI",     "openai_api_key"),
            ("DeepSeek",   "deepseek_api_key"),
            ("xAI (Grok)", "x_api_key"),
            ("Groq",       "groq_api_key"),
            ("Anthropic",  "anthropic_api_key"),
            ("Google AI",  "googleai_api_key"),
            ("Alibaba",    "alibaba_api_key"),
            ("OpenRouter", "openrouter_api_key"),
        ]
        for label, attr in providers_keys:
            if not getattr(ss, attr, None) or getattr(ss, attr) in ("missing", "miss", ""):
                val = st.text_input(f"{label} API Key", type="password", key=f"input_{attr}")
                if val:
                    setattr(ss, attr, val)

        provided = [lbl for lbl, attr in providers_keys
                    if getattr(ss, attr, None) not in (None, "missing", "miss", "")]
        if provided:
            st.success(f"✅ {', '.join(provided)}")
        else:
            st.warning("⚠️ No API keys provided")

    st.divider()

    # Model selector — grouped by provider
    st.subheader("🎯 AI Configuration")

    provider_names = list(models_by_provider.keys())
    # Detect current model and pre-select provider
    active_ci = ss.get("active_chat", 0)
    current_model = ss[chat_key(active_ci, "model_name")]
    default_provider = "Groq"
    for pname, pmodels in models_by_provider.items():
        if current_model in pmodels:
            default_provider = pname
            break

    selected_provider = st.selectbox(
        "🏢 Provider",
        provider_names,
        index=provider_names.index(default_provider),
        key="sidebar_provider"
    )
    provider_models = list(models_by_provider[selected_provider])

    # Custom model input — lets user type any model ID not in the list
    custom_model = st.text_input(
        "✏️ Custom model ID",
        placeholder="e.g. gpt-4o-2024-08-06",
        key="custom_model_input",
        help="Type any model ID not in the list above to use it directly"
    )
    if custom_model:
        if custom_model not in provider_models:
            provider_models = [custom_model] + provider_models
        default_model_idx = 0
    else:
        default_model_idx = provider_models.index(current_model) if current_model in provider_models else 0

    selected_model = st.selectbox(
        "🤖 Model",
        provider_models,
        index=default_model_idx,
        key="sidebar_model"
    )

    get_assistant = st.selectbox("👤 Assistant", assistant_list,
                                  index=assistant_list.index(ss[chat_key(active_ci, "assistant_name")]),
                                  key="sidebar_assistant")
    get_format    = st.selectbox("📝 Reply Format", format_list,
                                  index=format_list.index(ss[chat_key(active_ci, "format_name")]),
                                  key="sidebar_format")

    st.divider()

    # Options
    st.subheader("⚙️ Options")
    translate_in   = st.selectbox("🌐 Translate to", ["none", "English", "French", "Japanese", "Italian", "Spanish"])
    instructions   = st.text_input("📋 Additional Instructions")

    col1, col2 = st.columns(2)
    play_audio_     = col1.checkbox('🔊 Audio',    value=False)
    copy_reply_     = col2.checkbox('📋 Copy',     value=False)
    run_code        = col1.checkbox('⚡ Run Code', value=False)
    use_smol_agents = col2.checkbox('🤖 Agentic',  value=False)
    if use_smol_agents:
        st.info("🤖 Agentic AI: uses web search, code execution, and multi-step reasoning.")

    user_avi = st.selectbox('👤 Avatar', ['🧑🏻','🧔🏻','👩🏻','👧🏻','👸🏻','👱🏻‍♂️','🧑🏼','👸🏼','🧒🏽','👳🏽','👴🏼','🎅🏻'])

    st.divider()

    # ── Image Generation ───────────────────────────────────────────────────────
    st.subheader("🎨 Image Generation")

    _img_providers = ["OpenAI", "Google", "OpenRouter", "Alibaba"]
    _img_models_map = {
        "OpenAI":     list(OPENAI_IMAGE_MODELS.keys()),
        "Google":     list(GOOGLE_IMAGE_MODELS.keys()),
        "OpenRouter": list(OPENROUTER_IMAGE_MODELS.keys()),
        "Alibaba":    list(ALIBABA_IMAGE_MODELS.keys()),
    }

    if "img_provider" not in ss:
        ss["img_provider"] = "OpenRouter"
    if "img_model" not in ss:
        ss["img_model"] = DEFAULT_IMAGE_MODELS["OpenRouter"]
    if "img_aspect" not in ss:
        ss["img_aspect"] = "1:1"

    img_provider = st.selectbox(
        "🏢 Image Provider", _img_providers,
        index=_img_providers.index(ss["img_provider"]),
        key="img_provider_sel"
    )
    # Reset model when provider changes
    if img_provider != ss["img_provider"]:
        ss["img_provider"] = img_provider
        ss["img_model"] = DEFAULT_IMAGE_MODELS[img_provider]

    _cur_img_models = _img_models_map[img_provider]
    _cur_idx = _cur_img_models.index(ss["img_model"]) if ss["img_model"] in _cur_img_models else 0
    img_model = st.selectbox(
        "🖼️ Image Model", _cur_img_models,
        index=_cur_idx,
        key="img_model_sel"
    )
    ss["img_model"] = img_model

    img_aspect = st.selectbox(
        "📐 Aspect Ratio",
        ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
        index=["1:1","16:9","9:16","4:3","3:4","3:2","2:3"].index(ss["img_aspect"]),
        key="img_aspect_sel"
    )
    ss["img_aspect"] = img_aspect

    st.caption("💡 Trigger: `#create`, `#imagine`, `#draw`, `#paint`")

    st.divider()

    # File uploads
    st.subheader("📁 Attachments")
    uploaded_image = st.file_uploader("🖼️ Image", type=("jpg", "png", "jpeg"))
    uploaded_file  = st.file_uploader("📄 Text",  type=("txt", "md"))

    st.divider()

    # Chat management (for active chat)
    st.subheader("💬 Chat Management")
    col12, col22 = st.columns(2)
    if col12.button("🗑️ Clear"):
        ass_content = assistants[get_assistant] + features['reply_style'][get_format]
        ss[chat_key(active_ci, "messages")] = [{"role": "system", "content": ass_content}]
        st.rerun()
    if col22.button("🧹 Clear sys"):
        ss[chat_key(active_ci, "messages")] = [
            e for e in ss[chat_key(active_ci, "messages")] if e['role'] != 'system'
        ]
        st.rerun()

    # Save/Load
    st.subheader("💾 Save / Load")
    col_save, col_export = st.columns(2)
    with col_save:
        if st.button("💾 Save"):
            cname = save_chat_as_pickle(active_ci)
            st.success(f"Saved: {cname}")
    with col_export:
        if st.button("📤 Export"):
            md_export = export_chat_as_markdown(active_ci)
            st.download_button("⬇️ .md", data=md_export,
                               file_name=f"chat_{get_assistant}_{time.strftime('%Y%m%d_%H%M%S')}.md",
                               mime="text/markdown")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chats'))
    files_in_chats = os.listdir(base_dir) if os.path.exists(base_dir) else []
    if not files_in_chats:
        files_in_chats = ["No chats available"]
    chat_path_sel = st.selectbox("📂 Saved chats", files_in_chats)
    full_path = os.path.join('chats/', chat_path_sel)

    col_load, col_del = st.columns(2)
    if col_load.button("📥 Load"):
        if chat_path_sel != "No chats available":
            ss[chat_key(active_ci, "messages")] = load_chat_from_pickle(full_path)
            st.success("Loaded!")
        else:
            st.warning("Nothing to load")
    if col_del.button("🗑️ Delete"):
        if chat_path_sel != "No chats available":
            delete_file(full_path)
            st.success("Deleted!")
        else:
            st.warning("Nothing to delete")

    st.divider()

    # Info
    Info = st.button("📖 Help")
    st.markdown("- [OpenAI keys](https://platform.openai.com/account/api-keys)")
    st.markdown("- [Source code](https://github.com/johndef64/mychatbot)")

# ─── Apply sidebar selections to active chat ──────────────────────────────────
ss[chat_key(active_ci, "assistant_name")] = get_assistant
ss[chat_key(active_ci, "format_name")]    = get_format
ss[chat_key(active_ci, "model_name")]     = selected_model

ass_content = assistants[get_assistant] + features['reply_style'][get_format]
ss["assistant"] = ass_content
ss[chat_key(active_ci, "messages")] = update_assistant_in_chat(
    ss[chat_key(active_ci, "messages")], ass_content,
    ss[chat_key(active_ci, "sys_addings")]
)

# Apply instructions / file uploads
if instructions:
    ss[chat_key(active_ci, "sys_addings")].append(instructions)

if uploaded_file:
    text = uploaded_file.read().decode()
    ss[chat_key(active_ci, "messages")].append(
        {"role": "system", "content": "Read the text below and add it to your knowledge:\n\n" + text}
    )

# ─── Main area ────────────────────────────────────────────────────────────────
st.title("🤖 MyChatbot v2.0")
st.caption("Multi-AI Assistant · OpenAI · Anthropic · Google · Alibaba · DeepSeek · xAI · Groq")

# ─── Multi-chat tabs ──────────────────────────────────────────────────────────
# Chat tab controls
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
with ctrl_col1:
    if ss["num_chats"] < MAX_CHATS:
        if st.button("➕ New Chat"):
            new_i = ss["num_chats"]
            ss["num_chats"] += 1
            init_chat(new_i)
            ss["active_chat"] = new_i
            st.rerun()
with ctrl_col2:
    if ss["num_chats"] > 1:
        if st.button("✖ Close Chat"):
            close_i = ss["active_chat"]
            # Shift chats down
            for i in range(close_i, ss["num_chats"] - 1):
                for k in ["assistant_name","format_name","model_name","messages","sys_addings","reply","title"]:
                    ss[chat_key(i, k)] = ss[chat_key(i+1, k)]
            # Clear last
            last = ss["num_chats"] - 1
            for k in ["assistant_name","format_name","model_name","messages","sys_addings","reply","title"]:
                if chat_key(last, k) in ss:
                    del ss[chat_key(last, k)]
            ss["num_chats"] -= 1
            ss["active_chat"] = min(ss["active_chat"], ss["num_chats"] - 1)
            st.rerun()

# Build tab labels
tab_labels = [ss[chat_key(i, "title")] for i in range(ss["num_chats"])]
tabs = st.tabs(tab_labels)

# The active chat index drives everything; read it once here
ci = ss.get("active_chat", 0)
model       = ss[chat_key(ci, "model_name")]
voice       = voice_dict.get(ss[chat_key(ci, "assistant_name")], "echo")
chatbot_avi = avatar_dict.get(ss[chat_key(ci, "assistant_name")], "🤖")

# ── Chat input is declared OUTSIDE the tabs so Streamlit anchors it to the
#    bottom of the page, not in the middle of the tab content.
prompt = st.chat_input(
    "Type your message… (@ to clear, + to add without reply)",
    key="chat_input_global"
)

for tab_i, tab in enumerate(tabs):
    with tab:
        if tab_i != ss["active_chat"]:
            # Non-active tab: show switch button + last message preview
            if st.button(f"Switch here", key=f"switch_{tab_i}"):
                ss["active_chat"] = tab_i
                st.rerun()
            msgs_preview = ss[chat_key(tab_i, "messages")]
            non_sys = [m for m in msgs_preview if m['role'] != 'system']
            if non_sys:
                last_msg = non_sys[-1]
                preview = str(last_msg['content'])[:120] + ("…" if len(str(last_msg['content'])) > 120 else "")
                st.caption(f"**{last_msg['role']}**: {preview}")
            else:
                st.caption("*Empty chat*")
            continue

        # ── Active tab content ──────────────────────────────────────────────
        # Info bar
        info_col1, info_col2, info_col3 = st.columns([3, 2, 2])
        with info_col1:
            st.markdown(f"**Model:** `{model}`")
        with info_col2:
            st.markdown(f"**Assistant:** `{ss[chat_key(ci, 'assistant_name')]}`")
        with info_col3:
            new_title = st.text_input("Chat name", value=ss[chat_key(ci, "title")],
                                       key=f"title_input_{ci}", label_visibility="collapsed")
            ss[chat_key(ci, "title")] = new_title

        # Help
        if Info:
            st.success("🚀 Quick Commands")
            with st.expander("⚡ Commands", expanded=True):
                st.markdown("""
| Command | Action |
|---------|--------|
| `+message` | Add message without AI reply |
| `++instruction` | Add system instruction |
| `.` | Remove last message |
| `-` or `@` | Clear entire chat |
""")
            with st.expander("🤖 Assistants"):
                st.markdown("""
**Copilots**: base, creator, fixer, novelist, delamain, oracle, snake, roger
**Science**: leonardo, galileo, newton, mendel, watson, crick, venter, darwin, dawkins, penrose, turing
**Characters**: julia, mike, michael, miguel, francois, luca, hero, yoko, xiao, peng
**Languages**: english, french, italian, spanish, portuguese, korean, chinese, japanese
""")
        else:
            if f"hint_{ci}" not in ss:
                st.info("ℹ️ **Quick Commands**: `+message` add without reply · `++instruction` system prompt · `.` undo · `-` clear")
                ss[f"hint_{ci}"] = True

        # Display chat history
        messages = ss[chat_key(ci, "messages")]
        for msg in messages:
            if msg['role'] != 'system':
                if not isinstance(msg["content"], list):
                    avatar = user_avi if msg["role"] == 'user' else chatbot_avi
                    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# ── Handle prompt (outside tabs, anchored to bottom) ──────────────────────────
if prompt:
    messages = ss[chat_key(ci, "messages")]

    # ── Image generation trigger ───────────────────────────────────────────
    _is_img, _img_prompt = is_image_prompt(prompt)
    if _is_img:
        st.chat_message('user', avatar=user_avi).write(prompt)
        _img_provider = ss.get("img_provider", "OpenRouter")
        _img_model    = ss.get("img_model",    DEFAULT_IMAGE_MODELS["OpenRouter"])
        _img_aspect   = ss.get("img_aspect",   "1:1")

        with st.spinner(f"🎨 Generating image with {_img_model}…"):
            _img, _err = generate_image(
                prompt=_img_prompt,
                provider=_img_provider,
                model_label=_img_model,
                api_keys=load_api_keys(),
                aspect_ratio=_img_aspect,
            )

        if _err:
            st.error(f"❌ Image generation failed: {_err}")
            # Save error as assistant message so it's in history
            _err_msg = f"❌ Image generation error: {_err}"
            ss[chat_key(ci, "messages")].append({"role": "user",      "content": prompt})
            ss[chat_key(ci, "messages")].append({"role": "assistant", "content": _err_msg})
        else:
            # Save image to disk
            _img_path = save_image(_img, _img_prompt, _img_model, folder="images")
            # Show in chat
            with st.chat_message("assistant", avatar=chatbot_avi):
                st.image(_img, caption=f"🎨 {_img_model} · {_img_aspect} · {_img_prompt[:80]}")
                st.caption(f"💾 Saved: `{_img_path}`")
            # Store reference in chat history
            _rel = os.path.relpath(_img_path)
            ss[chat_key(ci, "messages")].append({"role": "user",      "content": prompt})
            ss[chat_key(ci, "messages")].append({
                "role": "assistant",
                "content": f"[Generated image: {_rel}]\n\nPrompt: _{_img_prompt}_\nModel: {_img_model} | Aspect: {_img_aspect}"
            })
        st.stop()

    if prompt in ["-", "@"]:
        ass_c = assistants[ss[chat_key(ci, "assistant_name")]] + features['reply_style'][ss[chat_key(ci, "format_name")]]
        ss[chat_key(ci, "messages")] = [{"role": "system", "content": ass_c}]
        st.balloons()
        time.sleep(0.5)
        st.rerun()

    elif prompt.startswith("+"):
        prompt = prompt[1:]
        role = "user"
        if prompt.startswith("+"):
            prompt = prompt[1:]
            role = "system"
        if role == "system":
            ss[chat_key(ci, "sys_addings")].append(prompt)
            st.success("✅ System instruction added!")
            time.sleep(0.8)
            st.rerun()
        else:
            ss[chat_key(ci, "messages")].append({"role": role, "content": prompt})
            st.chat_message(role, avatar=user_avi).write(prompt)
            st.info("📝 Message added without AI response")

    elif prompt in [".", "undo", "back"]:
        if len(messages) > 1:
            remove_last_non_system(ss[chat_key(ci, "messages")])
            st.success("↩️ Last message removed")
            time.sleep(0.7)
            st.rerun()
        else:
            st.warning("⚠️ Nothing to remove")

    else:
        if not ss.openai_api_key and not ss.groq_api_key and not ss.anthropic_api_key:
            st.info("Please add an API key in the sidebar.")
            st.stop()

        # Handle image upload
        image_path = None
        if uploaded_image:
            print('<Encoding Image...>')
            base64_image = encode_ioimage(uploaded_image)
            image_path = f"data:image/jpeg;base64,{base64_image}"
            image_add = {"role": 'user',
                         "content": [{"type": "image_url", "image_url": {"url": image_path}}]}
            if image_add not in ss[chat_key(ci, "messages")]:
                ss[chat_key(ci, "messages")].append(image_add)

        client = select_client(model)

        # Add user message
        ss[chat_key(ci, "messages")].append({"role": "user", "content": prompt})
        st.chat_message('user', avatar=user_avi).write(prompt)

        # Build thread (strip internal markers)
        chat_thread = []
        for msg in ss[chat_key(ci, "messages")]:
            if isinstance(msg["content"], list):
                chat_thread.append(msg)
            elif not str(msg["content"]).startswith('<<'):
                chat_thread.append(msg)

        # Generate reply
        try:
            if use_smol_agents:
                from utils import smol_agents, smol_research_task, smol_code_task, smol_web_search
                q = prompt.lower()
                web_kw  = ['search','find','research','current','latest','news','what is','who is']
                code_kw = ['code','python','script','function','algorithm','calculate','plot','graph']
                res_kw  = ['analyze','compare','study','report','summary','comprehensive']
                if any(k in q for k in code_kw):
                    st.info("🐍 Code Agent...")
                    reply = smol_code_task(prompt, model)
                elif any(k in q for k in res_kw):
                    st.info("🔬 Research Agent...")
                    reply = smol_research_task(prompt, model)
                elif any(k in q for k in web_kw):
                    st.info("🔍 Web Search Agent...")
                    reply = smol_web_search(prompt, model)
                else:
                    st.info("🤖 Multi-Tool Agent...")
                    reply = smol_agents(prompt, model)
                reply = f"🤖 **Agent Response**\n\n{reply}"
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=chat_thread,
                    max_tokens=get_max_tokens(model),
                    stream=False,
                )
                reply = response.choices[0].message.content

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Check your API key for the selected provider/model.")
            st.stop()

        reply, chain_of_thoughts = strip_think_tag(reply)
        if len(chain_of_thoughts) > 3:
            with st.expander("🧠 Chain of Thoughts", expanded=False):
                st.write(chain_of_thoughts)

        ss[chat_key(ci, "reply")] = reply

        # Remove image from context after use
        if uploaded_image:
            ss[chat_key(ci, "messages")] = [
                m for m in ss[chat_key(ci, "messages")]
                if not (isinstance(m.get("content"), list)
                        and any(p.get("type") == "image_url" for p in m["content"]))
            ]

        ss[chat_key(ci, "messages")].append({"role": "assistant", "content": reply})
        st.chat_message('assistant', avatar=chatbot_avi).write(reply)

        if check_copy_paste() and copy_reply_:
            pc.copy(ss[chat_key(ci, "reply")])

        if save_log:
            update_log(ss[chat_key(ci, "messages")][-2])
            update_log(ss[chat_key(ci, "messages")][-1])

        if translate_in != 'none':
            language = translate_in
            reply_language = rileva_lingua(reply)
            if reply_language == 'Japanese':
                translator = create_jap_translator(language)
            elif 'Chinese' in reply_language.split(" "):
                translator = create_chinese_translator(language)
            else:
                translator = create_translator(language)
            resp_t = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": translator},
                          {"role": "user", "content": reply}]
            )
            translation = "<<" + resp_t.choices[0].message.content + ">>"
            ss[chat_key(ci, "messages")].append({"role": "assistant", "content": translation})
            st.chat_message('assistant').write(translation)

        if play_audio_:
            Text2Speech(reply, voice=voice)

        if run_code:
            from ExecuteCode import ExecuteCode
            ExecuteCode(reply)
