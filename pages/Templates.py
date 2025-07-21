import streamlit as st
from mychatgpt import assistants, tasks

nested_dict = {
    "assistants":assistants,
    "tasks": {
        "reduction": tasks["reduction"],
        "details": "More details here"
    },
    "entry2": {
        "content": "Second entry text",
        "metadata": {"author": "UserX", "date": "2023-01-01"}
    },
    "entry3": {
        "main_text": "Third information block",
        "extras": ["list", "of", "items"]
    }
}

selected_entry = st.selectbox("Step 1: Select entry", list(nested_dict.keys()))
current_data = nested_dict[selected_entry]
print(selected_entry)

selected_subkey = st.selectbox("Step 2: Select sub-entry", list(current_data.keys()))
selected_value = current_data[selected_subkey]
print(selected_subkey)

st.info(f"{selected_value}")