# MyChatbot Configuration
import streamlit as st

# UI Configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF", 
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730"
    },
    "layout": {
        "sidebar_width": 350,
        "max_chat_width": 800
    },
    "animations": {
        "enable_balloons": True,
        "enable_snow": False,
        "enable_success_animations": True
    }
}

# Model Configuration
DEFAULT_MODELS = {
    "gpt": "gpt-4o-mini",
    "deepseek": "deepseek-r1-distill-llama-70b", 
    "anthropic": "claude-3-5-sonnet-latest",
    "grok": "grok-2-latest"
}

# Chat Configuration
CHAT_CONFIG = {
    "max_messages_history": 100,
    "auto_save": True,
    "auto_save_interval": 10,  # messages
    "enable_chain_of_thoughts": True,
    "enable_voice": True,
    "enable_copy": True
}

# Features Configuration  
FEATURES_CONFIG = {
    "enable_code_execution": True,
    "enable_file_upload": True,
    "enable_image_upload": True,
    "enable_translation": True,
    "enable_export": True,
    "enable_statistics": True
}

def load_user_config():
    """Load user-specific configuration from session state"""
    config = {}
    
    # Load from session state if available
    if "user_config" in st.session_state:
        config = st.session_state.user_config
    
    return config

def save_user_config(config):
    """Save user configuration to session state"""
    st.session_state.user_config = config

# Default assistant preferences
ASSISTANT_PREFERENCES = {
    "default_assistant": "none",
    "default_format": "base", 
    "favorite_assistants": ["delamain", "penrose", "leonardo"],
    "recent_assistants": []
}

# Quick commands help
QUICK_COMMANDS_HELP = {
    "@": "Clear entire chat",
    "-": "Clear entire chat (alternative)",
    "+message": "Add message without AI reply",
    "++instruction": "Add system instruction", 
    ".": "Remove last message pair",
    "undo": "Remove last message pair (alternative)",
    "back": "Remove last message pair (alternative)"
}
