import os, sys, contextlib, re, socket
import pickle
from openai import OpenAI
import streamlit as st
import base64
#from assistants import *
#from mychatgpt import GPT, play_audio
from utils import *
from assistants import *
# from mychatgpt import rileva_lingua, update_log

save_log=True
ss = st.session_state
### session states name definition ####
chat_num = '0'
assistant_name = f"assistant_{chat_num}"
format_name    = f"format_{chat_num}"
chat_n         = f"chat_{chat_num}"
sys_addings    = f"sys_add_{chat_num}"
model_name     = f"model_{chat_num}"
reply          = f"reply_{chat_num}"



def generate_chat_name(path, assistant_name):
    # Counter starts at 1
    index = 1
    # Continuously checking if the file with the current index exists
    while os.path.exists(os.path.join(path, f"{assistant_name}_{index}.pkl")):
        index += 1
    # Return the chat name with the next available index
    return f"{assistant_name}_{index}"

def save_chat_as_pickle(path='chats/'):
    if not os.path.exists(path):
        os.mkdir(path)

    chat_name = generate_chat_name(path, ss[assistant_name])
    # Save chat content in pickle format
    with open(os.path.join(path, chat_name + '.pkl'), 'wb') as file:
        pickle.dump(ss[chat_n], file)
    return chat_name

def load_chat_from_pickle(file_path):
    # Load chat content from pickle file
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    else:
        return False

def export_chat_as_markdown():
    """Export current chat as markdown format"""
    if len(ss[chat_n]) <= 1:
        return "# Empty Chat\n\nNo messages to export."
    
    markdown_content = f"# Chat Export - {get_assistant}\n\n"
    markdown_content += f"**Model:** {model}\n"
    markdown_content += f"**Assistant:** {get_assistant}\n"
    markdown_content += f"**Export Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "---\n\n"
    
    for i, msg in enumerate(ss[chat_n]):
        if msg['role'] == 'user':
            markdown_content += f"## 👤 User\n\n{msg['content']}\n\n"
        elif msg['role'] == 'assistant':
            markdown_content += f"## 🤖 {get_assistant}\n\n{msg['content']}\n\n"
        elif msg['role'] == 'system':
            markdown_content += f"*System: {msg['content']}*\n\n"
    
    return markdown_content

#%%


# Function to be executed on button click
def clearchat():
    ss[chat_n] = [{"role": "system", "content": ss['assistant']}]
    st.write("Chat cleared!")
def clearsys():
    ss[chat_n] = [entry for entry in ss[chat_n] if entry['role'] != 'system']
    st.write("System cleared!")

def remove_system_entries(input_list):
    return [entry for entry in input_list if entry.get('role') != 'system']
def update_assistant(input_list):
    updated_list = remove_system_entries(input_list)
    updated_list.append({"role": "system", "content": ss.assistant })
    for add in  ss[sys_addings]:
        updated_list.append({"role": "system", "content": add })
    return updated_list

def remove_last_non_system(input_list):
    # Iterate backwards to find and remove the last non-system entry
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i].get('role') != 'system':
            del input_list[i]  # Remove the entry
            break  # Exit the loop once the entry is removed
    return input_list



# Check if session state exists, if not, initialize it
if assistant_name not in ss:
    ss[assistant_name] = 'none'

format_list = list(features['reply_style'].keys())
if format_name not in ss:
    ss[format_name] = 'base'

if "assistant" not in ss:
    ss['assistant'] = assistants[ss[assistant_name]]

# Build assistant - this will be updated after user selection in sidebar


# assistant_list = list(assistants.keys())
assistant_list = [
    'none', 'base', 'creator', 'fixer', 'novelist', 'delamain',  'oracle', 'snake', 'roger', #'robert',
    'leonardo', 'galileo', 'newton',
    'mendel', 'watson', 'crick', 'venter',
    'collins', 'elsevier', 'springer',
    'darwin', 'dawkins',
    'penrose', 'turing', 'marker',
    'mike', 'michael', 'julia', 'jane', 'yoko', 'asuka', 'misa', 'hero', 'xiao', 'peng', 'miguel', 'francois', 'luca',
    'english', 'spanish', 'french', 'italian', 'portuguese', 'korean', 'chinese', 'japanese', 'japanese_teacher', 'portuguese_teacher'
]

# Try to import 'extra' from 'extra_assistant' if it's available
try:
    from extra_assistants import extra
except ImportError:
    # If the import fails, initialize 'extra' as an empty dictionary
    extra = {}
# Add values from 'extra' to 'assistants'
assistants.update(extra)
# Add keys from 'extra' to 'assistant_list'
assistant_list.extend(extra.keys())



# Initialize Chat Thread
if chat_n not in ss:
    #ss[chat_n] = [{"role": "assistant", "content": "How can I help you?"}]
    ss[chat_n] = [{"role": "system", "content": ss["assistant"]}]

if sys_addings not in ss:
    ss[sys_addings] = []

if model_name not in ss:
    ss[model_name] = "moonshotai/kimi-k2-instruct" #"deepseek-r1-distill-llama-70b" #"gpt-4o-mini"

if reply not in ss:
    ss[reply] = ""

# Update assistant in chat thread will be done after sidebar selection

# Immagazzina il valore corrente in session_state e imposta il valore predefinito se non esiste
# if 'assistant_index' not in ss:
#     ss.assistant_index = 0

#%%

# Check if the file exists
# if os.path.exists('openai_api_key.txt'):
if len(list(load_api_keys().keys() )) > 0:
    api_keys = load_api_keys()
    ss.openai_api_key   = api_keys.get("openai", "missing")
    ss.gemini_api_key   = api_keys.get("gemini", "missing")
    ss.deepseek_api_key = api_keys.get("deepseek", "missing")
    ss.x_api_key        = api_keys.get("grok", "missing")
    ss.groq_api_key     = api_keys.get("groq", "missing")
    ss.anthropic_api_key = api_keys.get("anthropic", "missing")

    # with open('openai_api_key.txt', 'r') as file:
    #     ss.openai_api_key = file.read().strip()
    #     #ss.openai_api_key = str(open('openai_api_key.txt', 'r').read())
else:
    ss.openai_api_key   = None
    ss.gemini_api_key   = None
    ss.deepseek_api_key = None
    ss.x_api_key        = None
    ss.groq_api_key    = None
    ss.anthropic_api_key = None

# Function to get local IP address
def get_local_ip():
    try:
        # Connect to a remote server to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        # Fallback to localhost if unable to determine IP
        return "localhost"

# Get the actual IP address
local_ip = get_local_ip()
streaming_url = f"http://{local_ip}:8501"

# Retro terminal print for the chatbot - only on first run
if 'app_initialized' not in ss:
    print("\n" + "#"*62)
    print("##" + " "*58 + "##")
    print("##      MYCHATBOT PRO - MULTI-AI ASSISTANT v1.0           ##")
    print("##" + " "*58 + "##")
    print("#"*62)
    print("|| STATUS: ONLINE")
    print(f"|| STREAMING: {streaming_url}")
    print("|| MODELS: OpenAI | DeepSeek | Grok | Groq | Anthropic")
    print("|| READY: Advanced AI capabilities loaded")
    print("-"*62)
    print("SERVER RUNNING... Access your chatbot via URL above")
    print("-"*62 + "\n")
    print(">>> App Ready!")
    
    # Mark app as initialized
    ss['app_initialized'] = True


# <<<<<<<<<<<<Sidebar code>>>>>>>>>>>>>
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Keys section
    with st.expander("🔑 API Keys", expanded=not any([ss.openai_api_key, ss.deepseek_api_key, ss.x_api_key, ss.groq_api_key, ss.anthropic_api_key])):
        if not ss.openai_api_key:
            ss.openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        if not ss.deepseek_api_key:
            ss.deepseek_api_key = st.text_input("Deepseek API Key",  type="password")
        if not ss.x_api_key:
            ss.x_api_key  = st.text_input("Xai API Key",  type="password")
        if not ss.groq_api_key:
            ss.groq_api_key  = st.text_input("Groq API Key",  type="password")
        if not ss.anthropic_api_key:
            ss.anthropic_api_key  = st.text_input("Anthropic API Key",  type="password")

        # API Status indicator
        keys = {
            "OpenAI": ss.openai_api_key,
            "DeepSeek": ss.deepseek_api_key,
            "X": ss.x_api_key,
            "Groq": ss.groq_api_key,
            "Anthropic": ss.anthropic_api_key
        }
        provided_keys = [name for name, key in keys.items() if key]
        if provided_keys:
            st.success(f"✅ {', '.join(provided_keys)} connected")
        else:
            st.warning("⚠️ No API keys provided")

    st.divider()
    
    # Model and Assistant section
    st.subheader("🎯 AI Configuration")
    model = st.selectbox('🤖 Model:', api_models, index=api_models.index(ss[model_name]))
    get_assistant = st.selectbox("👤 Assistant", assistant_list, index=assistant_list.index(ss[assistant_name]))
    get_format = st.selectbox("📝 Reply Format", format_list, index=format_list.index(ss[format_name]))

    st.divider()
    
    # Options section
    st.subheader("⚙️ Options")
    translate_in = st.selectbox("🌐 Translate to", ["none", "English", "French", "Japanese", "Italian", "Spanish"])
    instructions = st.text_input("📋 Additional Instructions")
    
    col1, col2 = st.columns(2)
    play_audio_ = col1.checkbox('🔊 Audio', value=False)
    copy_reply_ = col2.checkbox('📋 Copy', value=True)
    run_code = col1.checkbox('⚡ Run Code', value=False)
    # if col2.button("Copy Reply"):
    #     pc.copy(ss[reply])

    # Update session state with the selected value
    ss[assistant_name] = get_assistant
    ss[format_name] = get_format
    ss[model_name] = model
    
    # Build assistant with updated values
    ss['assistant'] = assistants[ss[assistant_name]] + features['reply_style'][ss[format_name]]
    
    # Update assistant in chat thread
    ss[chat_n] = update_assistant(ss[chat_n])
    
    st.divider()
    
    # File uploads section
    st.subheader("📁 File Uploads")
    
    image_path = None
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    uploaded_image = st.file_uploader("🖼️ Upload Image", type=("jpg", "png", "jpeg"))
    def encode_ioimage(uploaded_image):
        image_data = uploaded_image.read()
        return base64.b64encode(image_data).decode("utf-8")

    uploaded_file = st.file_uploader("📄 Upload Text", type=("txt", "md"))
    
    st.divider()
    
    # Chat management
    st.subheader("💬 Chat Management")
    col12, col22 = st.columns(2)
    if col12.button("🗑️ Clear chat"):
        clearchat()
    if col22.button("🧹 Clear system"):
        clearsys()
    
    user_avi = st.selectbox('👤 Your Avatar', ['🧑🏻', '🧔🏻', '👩🏻', '👧🏻', '👸🏻','👱🏻‍♂️','🧑🏼','👸🏼','🧒🏽','👳🏽','👴🏼', '🎅🏻', ])

    # Chat save/load section
    st.subheader("💾 Save/Load Chats")
    col_save, col_export = st.columns(2)
    
    with col_save:
        if st.button("💾 Save Chat"):
            chat_name = save_chat_as_pickle()
            st.success(f"✅ Chat saved as {chat_name}!")
    
    with col_export:
        if st.button("📄 Export MD"):
            markdown_export = export_chat_as_markdown()
            st.download_button(
                label="⬇️ Download",
                data=markdown_export,
                file_name=f"chat_export_{get_assistant}_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

    # List files in the 'chats/' directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chats'))
    files_in_chats = os.listdir(base_dir) if os.path.exists(base_dir) else (os.makedirs(base_dir, exist_ok=True), [])[1]
    if len(files_in_chats) == 0:
        files_in_chats = ["No chats available"]

    chat_path = st.selectbox("📂 Available chats", files_in_chats)
    full_path = os.path.join('chats/', chat_path)

    col1, col2 = st.columns(2)
    if col1.button("📥 Load"):
        if chat_path != "No chats available":
            ss[chat_n] = load_chat_from_pickle(full_path)
            st.success("✅ Chat loaded!")
        else:
            st.warning("⚠️ No chats to load")

    if col2.button("🗑️ Delete"):
        if chat_path != "No chats available":
            delete_file(full_path)
            st.success("✅ Chat deleted!")
        else:
            st.warning("⚠️ No chats to delete")
    
    st.divider()
    
    # Info and links
    st.subheader("ℹ️ Information")
    Info = st.button("📖 Show Help")
    
    st.markdown("🔗 **Useful Links:**")
    st.markdown("- [Get OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("- [View source code](https://github.com/johndef64/mychatbot)")

############################################################################################
############################################################################################

# from mychatgpt import gpt_models, deepseek_models, x_models, groq_models, Groq
from groq import Groq


# selct client
def select_client(model):
    if model in gpt_models:
        client = OpenAI(api_key=load_api_keys()["openai"])
    elif model in deepseek_models:
        print("using DeepSeek model")
        client = OpenAI(api_key=load_api_keys()["deepseek"], base_url="https://api.deepseek.com")
    # elif model in x_models:
    elif "grok" in model:
        print("using Xai model")
        client = OpenAI(api_key=load_api_keys()["grok"], base_url="https://api.x.ai/v1")
    elif model in groq_models: 
        print("using Groq models")
        client = Groq(api_key=load_api_keys()["groq"])
    elif model in anthropic_models:
        print("using Anthorpic models")
        client = OpenAI(api_key=load_api_keys()["anthropic"],base_url="https://api.anthropic.com/v1")
    return client


# <<<<<<<<<<<< >>>>>>>>>>>>>

def add_instructions(instructions):
    if not any(entry.get("role") == "system" and instructions in entry.get("content", "") 
           for entry in ss[chat_n]):
        ss[chat_n].append({"role": "system", "content": instructions})

### Add Context to system
if instructions:
   #add_instructions(instructions)
   ss[sys_addings].append(instructions)

if uploaded_file:
    text = uploaded_file.read().decode()
    ss[chat_n].append({"role": "system", "content": "Read the text below and add it's content to your knowledge:\n\n"+text})

#if uploaded_file:
#image = uploaded_image.read().decode()
#ss[chat_n] = ...

# # Update session state with the selected value
# ss[assistant_name] = get_assistant
# ss["format_name"] = get_format


# <<<<<<<<<<<<Display header>>>>>>>>>>>>>

st.title("🤖 MyChatbot v1.0")
st.caption("🚀 Multi-AI Assistant powered by OpenAI, DeepSeek, Grok, Groq & Anthropic")

# Add a nice header with model info
# Model status indicator
model_status_map = {
    "gpt": "🟢 OpenAI",
    "deepseek": "🔵 DeepSeek", 
    "grok": "🟡 xAI",
    "claude": "🟣 Anthropic",
    "llama": "🟠 Meta",
    "gemma": "🔴 Google",
    "mistral": "⚪ Mistral"
}

model_provider = "❓ Unknown"
for key, value in model_status_map.items():
    if key in model.lower():
        model_provider = value
        break

col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown(f"**Current Model:** `{model}` ({model_provider})")
with col2:
    st.markdown(f"**Assistant:** `{get_assistant}`")
with col3:
    st.markdown(f"**Format:** `{get_format}`")

st.divider()

# Draft information formatted within an info box
info = """#### Quick Commands:
- Start message with "+" to add message without getting a reply
- Start message with "++" to add additional system instructions
- Enter ":" to pop out last iteration
- Enter '-' to clear chat
"""
# Display the info box using Streamlit's 'info' function
# st.info(info)

AssistantInfo = """
#### Copilots 💻
- **Base**: Assists with basic tasks and functionalities.
- **Novelist**: Specializes in creative writing assistance.
- **Creator**: Aids in content creation and ideation.
- **Fixer**: Can fix any text based on the context.
- **Delamain**: Coding copilot for every purpose.
- **Oracle**: Coding copilot for every purpose.
- **Snake**: Python coding copilot for every purpose.
- **Roger**: R coding copilot for every purpose.

#### Scientific 🔬
- **Leonardo**: Supports scientific research activities.
- **Newton**: Aids in Python-based scientific computations (Python).
- **Galileo**: Specializes in scientific documentation (Markdown).
- **Mendel**: Assists with data-related scientific tasks.
- **Watson**: Focuses on typesetting scientific documents (LaTeX).
- **Venter**: Supports bioinformatics and coding tasks (Python).
- **Crick**: Specializes in structuring scientific content (Markdown).
- **Darwin**: Aids in evolutionary biology research tasks.
- **Dawkins**: Supports documentation and writing tasks (Markdown).
- **Penrose**: Assists in theoretical research fields.
- **Turing**: Focuses on computational and AI tasks (Python).
- **Marker**: Specializes in scientific documentation (Markdown).
- **Collins**: Aids in collaborative scientific projects.
- **Elsevier**: Focuses on publication-ready document creation (LaTeX).
- **Springer**: Specializes in academic content formatting (Markdown).

#### Characters 🎭
- **Julia**: Provides character-based creative support.
- **Mike**: Provides character-based interactive chat.
- **Michael**: Provides character-based interactive chat (English).
- **Miguel**: Provides character-based interactive chat (Portuguese).
- **Francois**: Provides character-based interactive chat (French).
- **Luca**: Provides character-based interactive chat (Italian).
- **Hero**: Provides character-based interactive chat (Japanese).
- **Yoko**: Provides character-based creative support (Japanese).
- **Xiao**: Provides character-based creative support (Chinese).
- **Peng**: Provides character-based interactive chat (Chinese).

#### Languages 🌐
- **English, French, Italian, Portuguese**
- **Chinese**: Facilitates Chinese language learning.
- **Japanese**: Aids in Japanese language learning and translation.
- **Japanese Teacher**: Specializes in teaching Japanese.
- **Portuguese Teacher**: Provides assistance with learning Portuguese.

"""

# Quick Commands info box with better formatting
if Info:
    st.success("🚀 **Quick Commands Guide**")
    
    with st.expander("⚡ Command Shortcuts", expanded=True):
        st.markdown("""
        | Command | Action |
        |---------|--------|
        | `+message` | Add message without AI reply |
        | `++instruction` | Add system instruction |
        | `.` | Remove last message  |
        | `-` or `@` | Clear entire chat |
        """)
    
    with st.expander("🤖 Available Assistants", expanded=False):
        st.markdown(AssistantInfo)
        
    with st.expander("🎯 Pro Tips", expanded=False):
        st.markdown("""
        - **Multi-modal**: Upload images and text files for context
        - **Translation**: Automatic translation to any supported language
        - **Voice**: Enable text-to-speech for responses
        - **Code Execution**: Run Python code directly from responses
        - **Chat Management**: Save/load conversations for later use
        """)
else:
    # Show compact help hint
    if "header_info" not in ss:
        st.info("ℹ️ **Quick Commands**: Type `+message` to add without reply, `++instruction` for system prompts, `.` to remove last message, `-` or `@` to clear chat")
        ss.header_info = True
# <<<<<<<<<<<< >>>>>>>>>>>>>


# Update Language Automatically
#if ss.persona not in ss[chat_n]:
#    ss[chat_n] = update_assistant(ss[chat_n])

voice = voice_dict.get(get_assistant, "echo")
chatbot_avi = avatar_dict.get(get_assistant, "🤖")

print("Voice:", voice)



# Trigger the specific function based on the selection
#if assistant and not ss[chat_n] == [{"role": "system", "content": assistants[assistant]}]:
#    ss[chat_n] = [{"role": "system", "content": assistants[assistant]}]
#    #st.write('assistant changed')


# <<<<<<<<<<<<Display chat>>>>>>>>>>>>>
def display_chat():
    for msg in ss[chat_n]:
        if msg['role'] != 'system':
            if not isinstance(msg["content"], list):
                # Avatar
                if msg["role"] == 'user':
                    avatar = user_avi
                else:
                    avatar = chatbot_avi
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

display_chat()


def strip_think_tag(input_string):
    # Trova la parte all'interno dei tag <think>
    pattern = r"<think>(.*?)</think>"
    think_part = re.findall(pattern, input_string, re.DOTALL)
    # Rimuovi la parte <think> dalla stringa originale
    senza_think = re.sub(pattern, '', input_string, flags=re.DOTALL).strip()
    # Rimuovi eventuali spazi aggiuntivi
    think_part = think_part[0].strip() if think_part else ''
    return senza_think, think_part

# <<<<<<<<<<<<Engage chat>>>>>>>>>>>>>

# Enhanced quick commands with better UX
if prompt := st.chat_input("Type your message here... (use @ to clear, + to add without reply)"):
    # Quick commands with better feedback:
    if prompt in ["-", "@"]:
        clearchat()
        st.balloons()  # Fun feedback
        time.sleep(0.7)
        st.rerun()

    elif prompt.startswith("+"):
        prompt = prompt[1:]
        role = "user"
        if prompt.startswith("+"):
            prompt = prompt[1:]
            role = "system"

        if role == "system":
            ss[sys_addings].append(prompt)
            st.success("✅ System instruction added!")
            time.sleep(1)
            st.rerun()
        else:
            ss[chat_n].append({"role": role, "content": prompt})
            st.chat_message(role, avatar=user_avi).write(prompt)
            st.info("📝 Message added without AI response")

    elif prompt in [".", "undo", "back"]:
        if len(ss[chat_n]) > 1:
            remove_last_non_system(ss[chat_n])
            st.success("↩️ Last message pair removed")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("⚠️ No messages to remove")

    else:
        if not ss.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            

        if image_path or uploaded_image:
            # if image_path:
            #     if image_path.startswith('http'):
            #         print('<Image path:',image_path, '>')
            #     pass
            # else:
            print('<Enconding Image...>')
            if uploaded_image:
                base64_image = encode_ioimage(uploaded_image)
            else:
                base64_image = encode_image(image_path)

                
            image_path = f"data:image/jpeg;base64,{base64_image}"

            image_add = {"role": 'user',
                        "content": [{"type": "image_url", "image_url": {"url": image_path} }] }
            if image_add not in ss[chat_n]:
                ss[chat_n].append(image_add)

        #client = OpenAI(api_key=ss.openai_api_key)
        client = select_client(model)
        
        # change model if model is not multimodal
        if image_path:
            if model in gpt_models:
                pass
            elif model in x_models:
                model = "grok-2-vision-1212"
        
        # Get User Prompt:
        ss[chat_n].append({"role": "user", "content": prompt})
        st.chat_message('user', avatar=user_avi).write(prompt)
        
        # Build Chat Thread
        chat_thread = []
        for msg in ss[chat_n]:
            if isinstance(msg["content"], list):
                chat_thread.append(msg)
            elif not msg["content"].startswith('<<'):
                chat_thread.append(msg)




        # Generate Reply        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=chat_thread,
                max_tokens=get_max_tokens(model),
                stream=False,
                #top_p=1,
                #frequency_penalty=0,
                #presence_penalty=0
            )

            reply = response.choices[0].message.content
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("💡 Try switching to a different model or check your API keys")
            st.stop()

        reply, chain_of_thoughts = strip_think_tag(reply)
        if len(chain_of_thoughts) > 3:
            with st.expander("🧠 Chain of Thoughts", expanded=False):
                st.write(chain_of_thoughts)
        ss[reply] = reply

        # Opt out image from context
        if uploaded_image:
            # Sostituisci se l'ultimo messaggio è multimodale
            # Filtra e rimuovi tutti i messaggi che contengono una parte con tipo "image_url"
            ss[chat_n] = [
                msg for msg in ss[chat_n]
                if not (
                    isinstance(msg.get("content"), list)
                    and any(part.get("type") == "image_url" for part in msg["content"])
                )
            ]
            print(ss[chat_n])


        # Append Reply
        ss[chat_n].append({"role": "assistant", "content": reply})
        st.chat_message('assistant', avatar=chatbot_avi).write(reply)
        

        if check_copy_paste() and copy_reply_:
            pc.copy(ss[reply])

        if save_log:
            update_log(ss[chat_n][-2])
            update_log(ss[chat_n][-1])

        if translate_in != 'none':
            language = translate_in
            reply_language = rileva_lingua(reply)
            if reply_language == 'Japanese':
                translator = create_jap_translator(language)
            elif 'Chinese' in reply_language.split(" "):
                translator = create_chinese_translator(language)
            else:
                translator = create_translator(language)
            response_ = client.chat.completions.create(model=model,
                                                    messages=[{"role": "system", "content": translator},
                                                                {"role": "user", "content": reply}])
            translation = "<<"+response_.choices[0].message.content+">>"
            ss[chat_n].append({"role": "assistant", "content": translation})
            st.chat_message('assistant').write(translation)


        if play_audio_:
            Text2Speech(reply, voice=voice)

        if run_code:
            from ExecuteCode import ExecuteCode
            ExecuteCode(reply)



    #with col2:
    #    if st.button("New Chat"):
    #    clearchat()






#%%

# Chat statistics
print_stats = False
if len(ss[chat_n]) > 1 and print_stats:
    user_messages = len([msg for msg in ss[chat_n] if msg['role'] == 'user'])
    assistant_messages = len([msg for msg in ss[chat_n] if msg['role'] == 'assistant'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Total Messages", user_messages + assistant_messages)
    with col2:
        st.metric("👤 Your Messages", user_messages)
    with col3:
        st.metric("🤖 AI Responses", assistant_messages)
    with col4:
        if assistant_messages > 0:
            avg_length = sum(len(msg['content']) for msg in ss[chat_n] if msg['role'] == 'assistant') // assistant_messages
            st.metric("📊 Avg Response Length", f"{avg_length} chars")
    
    st.divider()

def export_chat_as_markdown():
    """Export current chat as markdown format"""
    if len(ss[chat_n]) <= 1:
        return "# Empty Chat\n\nNo messages to export."
    
    markdown_content = f"# Chat Export - {get_assistant}\n\n"
    markdown_content += f"**Model:** {model}\n"
    markdown_content += f"**Assistant:** {get_assistant}\n"
    markdown_content += f"**Export Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "---\n\n"
    
    for i, msg in enumerate(ss[chat_n]):
        if msg['role'] == 'user':
            markdown_content += f"## 👤 User\n\n{msg['content']}\n\n"
        elif msg['role'] == 'assistant':
            markdown_content += f"## 🤖 {get_assistant}\n\n{msg['content']}\n\n"
        elif msg['role'] == 'system':
            markdown_content += f"*System: {msg['content']}*\n\n"
    
    return markdown_content

# Add a button to export chat
# st.caption("📤 Export Chat")
if st.button("📤 Export Chat"):
    markdown = export_chat_as_markdown()
    # Create a download link for the user
    st.download_button(
        label="Download Markdown",
        data=markdown,
        file_name="chat_export.md",
        mime="text/markdown",
    )


