# Enhanced utilities for MyChatbot
import streamlit as st
import time
import json
from datetime import datetime

def show_typing_indicator():
    """Show a typing indicator animation"""
    placeholder = st.empty()
    for i in range(3):
        placeholder.text("🤖 Typing" + "." * (i + 1))
        time.sleep(0.5)
    placeholder.empty()

def format_message_time(timestamp=None):
    """Format message timestamp"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%H:%M")

def get_chat_statistics(chat_messages):
    """Calculate chat statistics"""
    stats = {
        "total_messages": 0,
        "user_messages": 0,
        "assistant_messages": 0,
        "total_chars": 0,
        "avg_response_length": 0,
        "conversation_duration": None
    }
    
    if not chat_messages or len(chat_messages) <= 1:
        return stats
    
    for msg in chat_messages:
        if msg.get('role') != 'system':
            stats["total_messages"] += 1
            if msg.get('role') == 'user':
                stats["user_messages"] += 1
            elif msg.get('role') == 'assistant':
                stats["assistant_messages"] += 1
                stats["total_chars"] += len(msg.get('content', ''))
    
    if stats["assistant_messages"] > 0:
        stats["avg_response_length"] = stats["total_chars"] // stats["assistant_messages"]
    
    return stats

def create_progress_bar(current, total, label="Progress"):
    """Create a progress bar with label"""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label}: {current}/{total}")

def show_model_info(model_name):
    """Show information about the selected model"""
    model_info = {
        "gpt-4o": {"provider": "OpenAI", "type": "Multimodal", "context": "128k"},
        "gpt-4o-mini": {"provider": "OpenAI", "type": "Fast", "context": "128k"},
        "deepseek-r1": {"provider": "DeepSeek", "type": "Reasoning", "context": "64k"},
        "claude-3-5-sonnet": {"provider": "Anthropic", "type": "Advanced", "context": "200k"},
        "grok-2": {"provider": "xAI", "type": "Conversational", "context": "128k"}
    }
    
    # Find matching model info
    info = None
    for key, value in model_info.items():
        if key in model_name.lower():
            info = value
            break
    
    if info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Provider: {info['provider']}")
        with col2:
            st.caption(f"Type: {info['type']}")
        with col3:
            st.caption(f"Context: {info['context']}")

def create_message_bubble(role, content, avatar, timestamp=None):
    """Create a styled message bubble"""
    with st.chat_message(role, avatar=avatar):
        if timestamp:
            st.caption(format_message_time(timestamp))
        st.write(content)

def export_chat_settings():
    """Export current chat settings as JSON"""
    settings = {
        "model": st.session_state.get("model_0", ""),
        "assistant": st.session_state.get("assistant_0", ""),
        "format": st.session_state.get("format_0", ""),
        "export_timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }
    return json.dumps(settings, indent=2)

def validate_api_key(api_key, provider):
    """Basic API key validation"""
    if not api_key:
        return False, "API key is required"
    
    min_lengths = {
        "openai": 50,
        "deepseek": 30,
        "anthropic": 40,
        "grok": 30,
        "groq": 30
    }
    
    min_length = min_lengths.get(provider.lower(), 20)
    if len(api_key) < min_length:
        return False, f"API key seems too short for {provider}"
    
    return True, "API key format looks valid"

def create_assistant_card(assistant_name, assistant_info):
    """Create a card display for assistant information"""
    with st.container():
        st.markdown(f"### {assistant_name.title()}")
        st.markdown(assistant_info.get('description', 'No description available'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Category: {assistant_info.get('category', 'General')}")
        with col2:
            st.caption(f"Language: {assistant_info.get('language', 'English')}")

def get_model_emoji(model_name):
    """Get emoji for model provider"""
    model_emojis = {
        "gpt": "🤖",
        "deepseek": "🧠", 
        "claude": "🎭",
        "grok": "🚀",
        "llama": "🦙",
        "gemma": "💎",
        "mistral": "🌪️"
    }
    
    for key, emoji in model_emojis.items():
        if key in model_name.lower():
            return emoji
    return "🤖"
