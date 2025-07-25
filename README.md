# 🤖 MyChatbot Pro v1.0 - Multi-AI Assistant

A powerful Streamlit-based chatbot application with support for multiple AI providers and specialized assistants.

## ✨ Features

### 🤖 Multi-AI Provider Support
- **OpenAI**: GPT-4o, GPT-4o-mini
- **DeepSeek**: Advanced reasoning models
- **xAI**: Grok-2, Grok-3, Grok-4
- **Anthropic**: Claude Sonnet, Opus, Haiku
- **Groq**: High-speed inference
- **Meta**: Llama models

### 👥 Specialized Assistants
- **💻 Copilots**: Base, Novelist, Creator, Fixer, Delamain, Oracle, Snake, Roger
- **🔬 Scientific**: Leonardo, Newton, Galileo, Mendel, Watson, Venter, Crick, Darwin, Dawkins, Penrose, Turing
- **🎭 Characters**: Julia, Mike, Michael, Miguel, Francois, Luca, Hero, Yoko, Xiao, Peng
- **🌐 Language Teachers**: Multi-language support with specialized tutors

### 🚀 Advanced Features
- **📷 Multi-modal**: Image and text file uploads
- **🔊 Audio**: Text-to-speech with voice selection
- **🌍 Translation**: Real-time translation to multiple languages
- **💾 Chat Management**: Save, load, and export conversations
- **⚡ Quick Commands**: Shortcuts for common actions
- **📊 Statistics**: Real-time chat analytics
- **🧠 Chain of Thoughts**: View AI reasoning process
- **💻 Code Execution**: Run Python code directly

## 🎯 Quick Commands

| Command | Action |
|---------|--------|
| `@` or `-` | Clear entire chat |
| `+message` | Add message without AI reply |
| `++instruction` | Add system instruction |
| `.`, `undo`, `back` | Remove last message pair |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Quick Start
```bash
# Clone the repository
git clone https://github.com/johndef64/mychatgpt.git
cd mychatgpt/mychatbot

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run MyChatbot.py
```

### API Keys Setup
Create an `api_keys.json` file with your API keys:
```json
{
    "openai": "your-openai-api-key",
    "deepseek": "your-deepseek-api-key", 
    "grok": "your-xai-api-key",
    "groq": "your-groq-api-key",
    "anthropic": "your-anthropic-api-key"
}
```

## 🎨 User Interface

### Main Features
- **Clean Header**: Model status and current configuration
- **Chat Statistics**: Real-time message analytics
- **Enhanced Sidebar**: Organized configuration sections
- **Export Options**: Save chats as Markdown or Pickle

### Improvements Made
- ✅ Better error handling with user-friendly messages
- ✅ Organized sidebar with expandable sections
- ✅ Real-time chat statistics
- ✅ Enhanced quick commands with feedback
- ✅ Model provider indicators
- ✅ Chain of thoughts visualization
- ✅ Export to Markdown functionality
- ✅ Improved API key management

## 🔧 Configuration

The app includes configurable options for:
- UI themes and colors
- Default models per provider
- Chat behavior settings
- Feature toggles
- Assistant preferences

## 📱 Usage Tips

1. **Multi-Provider**: Switch between AI providers based on your needs
2. **Specialized Tasks**: Choose appropriate assistants for specific domains
3. **File Context**: Upload images or text files for enhanced context
4. **Chat Management**: Save important conversations for later reference
5. **Quick Actions**: Use command shortcuts for faster interaction

## 🔗 API Key Sources

- [OpenAI API Keys](https://platform.openai.com/account/api-keys)
- [DeepSeek API Keys](https://platform.deepseek.com/api_keys)
- [xAI API Keys](https://console.x.ai/)
- [Groq API Keys](https://console.groq.com/keys)
- [Anthropic API Keys](https://console.anthropic.com/)

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
