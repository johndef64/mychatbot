# 🤖 MyChatbot Pro v2.0 - Multi-AI Assistant

A powerful Streamlit-based chatbot with support for multiple AI providers, multi-chat tabs, image generation, light/dark theme, and a dedicated Models Manager page.

## ✨ Features

### 🤖 Multi-AI Provider Support
- **OpenAI**: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, o1-mini
- **Anthropic**: Claude Opus 4.5/4.0, Sonnet 4.5/4.0, Haiku 3.5
- **Google**: Gemini 2.5 Pro/Flash, Gemini 2.0 Flash, Gemini 1.5 Pro/Flash
- **Alibaba**: Qwen-Max, Qwen-Plus, Qwen-Turbo, Qwen3 (235B/32B), QwQ-32B, Qwen-Coder
- **DeepSeek**: DeepSeek-Chat (V3), DeepSeek-Reasoner (R1)
- **xAI**: Grok-4, Grok-3, Grok-3-mini, Grok-2
- **Groq**: Llama-4 Maverick/Scout, Kimi-K2, QwQ-32B, DeepSeek-R1, Mistral Saba
- **OpenRouter**: 300+ models aggregated

### 💬 Multi-Chat Tabs
- Up to **6 simultaneous conversations** in browser tabs
- Each chat has its own model, assistant, history, and editable title
- `➕ New Chat` / `✖ Close Chat` controls
- Chat input always anchored to the bottom of the page

### 🎨 Image Generation
Trigger with magic words: **`#create`**, **`#imagine`**, **`#draw`**, **`#paint`**, **`#generate`**, **`#image`**

| Provider | Models |
|---|---|
| OpenAI | DALL-E 2, DALL-E 3, gpt-image-1 |
| Google | Gemini 2.5 Flash Image, Gemini 3.1 Flash, Imagen 4.0 Fast/Standard/Ultra |
| OpenRouter | Flux.2 Klein/Flex/Pro/Max, Seedream 4.5, RiverFlow v2 |
| Alibaba | WanX 2.0/2.1, Wan 2.2/2.5/2.6/2.7, Qwen-Image, Z-Image Turbo |

Images are saved to the `images/` folder with embedded metadata (prompt, model, timestamp).

### 🤖 Models Manager Page (`pages/Models.py`)
- Browse all models organized by provider
- **Fetch latest** button queries live provider APIs to discover new models
- Live pricing from OpenRouter (300+ models with $/1M token costs)
- Static pricing reference for all other providers
- Columns: Model · Context · Input $/1M · Output $/1M · Pricing source · Status

### 🌙 Light / Dark Theme
Toggle with the 🌙/☀️ button in the sidebar. Theme persists across the session.

### 👥 Specialized Assistants
- **💻 Copilots**: Base, Novelist, Creator, Fixer, Delamain, Oracle, Snake (Python), Roger (R)
- **🔬 Scientific**: Leonardo, Newton, Galileo, Mendel, Watson, Venter, Crick, Darwin, Dawkins, Penrose, Turing
- **🎭 Characters**: Julia, Mike, Michael, Miguel, Francois, Luca, Hero, Yoko, Xiao, Peng
- **🌐 Language Teachers**: English, French, Italian, Spanish, Portuguese, Korean, Chinese, Japanese

### 🚀 Additional Features
- **📷 Multi-modal**: Image and text file uploads as context
- **🔊 Text-to-Speech**: Voice selection per assistant
- **🌍 Translation**: Real-time translation to any supported language
- **💾 Chat Management**: Save (pickle), load, export as Markdown
- **🧠 Chain of Thoughts**: Expandable reasoning trace (DeepSeek-R1, QwQ, etc.)
- **⚡ Agentic AI**: Smol Agents mode with web search and code execution
- **💻 Code Execution**: Run Python code blocks directly from responses
- **✏️ Custom model ID**: Type any model not in the list to use it directly

## 🎯 Quick Commands

| Command | Action |
|---------|--------|
| `@` or `-` | Clear entire chat |
| `+message` | Add message without AI reply |
| `++instruction` | Add system instruction |
| `.`, `undo`, `back` | Remove last message |
| `#create <prompt>` | Generate image (also: `#imagine`, `#draw`, `#paint`) |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Quick Start
```bash
git clone https://github.com/johndef64/mychatbot.git
cd mychatbot

pip install -r requirements.txt

streamlit run MyChatbot.py
```

### API Keys — `api_keys.json`
```json
{
    "openai":     "sk-...",
    "anthropic":  "sk-ant-...",
    "googleai":   "AIza...",
    "alibaba":    "sk-...",
    "deepseek":   "sk-...",
    "grok":       "xai-...",
    "groq":       "gsk_...",
    "openrouter": "sk-or-..."
}
```

The file is loaded automatically at startup. Keys can also be entered at runtime in the sidebar.

### Optional dependency — Google image generation
```bash
pip install google-genai
```

## 📁 Project Structure

```
mychatbot/
├── MyChatbot.py          # Main app
├── utils.py              # Models, clients, helpers
├── image_gen.py          # Image generation (OpenAI/Google/OpenRouter/Alibaba)
├── assistants.py         # Assistant prompts
├── api_keys.json         # Your API keys (not committed)
├── pages/
│   └── Models.py         # Models Manager page
├── images/               # Generated images (auto-created)
├── chats/                # Saved chat pickles (auto-created)
└── reference_code/       # Reference scripts
```

## 🔗 API Key Sources

| Provider | URL |
|---|---|
| OpenAI | https://platform.openai.com/account/api-keys |
| Anthropic | https://console.anthropic.com/keys |
| Google AI | https://aistudio.google.com/app/apikey |
| Alibaba DashScope | https://dashscope.console.aliyun.com |
| DeepSeek | https://platform.deepseek.com/api_keys |
| xAI | https://console.x.ai/ |
| Groq | https://console.groq.com/keys |
| OpenRouter | https://openrouter.ai/keys |

## 📄 License

MIT License
