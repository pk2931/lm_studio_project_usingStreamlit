import streamlit as st
from PIL import Image
import pytesseract
from transformers import DonutProcessor, VisionEncoderDecoderModel
from openai import OpenAI
import re
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Ask your questions",
    page_icon="❓",
)

st.sidebar.success("Ask your questions")

user_question = st.text_input('', placeholder='Enter your question')

@st.cache_resource
def load_models():
    # Initialize models once using session state
    st.session_state.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    client = st.session_state.client
    return client

def postprocess_receipt(data):
    receipt_str = ""

    # Iterate over the keys in the dictionary
    for section, section_data in data.items():
        # Format 'menu' or 'items' sections with list of items
        if isinstance(section_data, list):
            receipt_str += f"\n{section.capitalize()}:\n"
            for item in section_data:
                item_str = ""
                for key, value in item.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (sub-items)
                        sub_items = " ".join(f"{k}: {v}" for k, v in value.items())
                        item_str += f"{key}: {sub_items} | "
                    else:
                        item_str += f"{key}: {value} | "
                receipt_str += f" - {item_str.strip(' | ')}\n"
        
        # Format subtotal and total sections
        elif isinstance(section_data, dict):
            receipt_str += f"\n{section.capitalize()}:\n"
            for sub_key, sub_value in section_data.items():
                receipt_str += f" {sub_key}: {sub_value}\n"
        
        # Handle other sections (e.g., cashprice, changeprice)
        elif isinstance(section_data, str):
            receipt_str += f"\n{section.capitalize()}:\n{section_data}"
        
        receipt_str = re.sub(r'[^A-Za-z0-9\s]', '', receipt_str)

    return receipt_str.strip()

# Function to interact with LM Studio via Mistral
def mistral_interaction(user_input):
    print("FORMATTED STRING: \n", user_input)
    streamed_completion = client.chat.completions.create(
        model="TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q4_K_S.gguf",
        messages=[{"role": "user", "content": user_input}],
        stream=True,
        temperature=0.7,  # Lower values reduce randomness and repetition
        top_p=0.9         # Limits token selection to a narrower subset
    )
    full_response = ""
    line_buffer = ""
    for chunk in streamed_completion:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            line_buffer += delta_content
            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                full_response += '\n'.join(lines[:-1])
                line_buffer = lines[-1]
    if line_buffer:
        full_response += line_buffer
        
    print("FINAL OUTPUT BY LLM: \n", full_response)

    return full_response

client = load_models()

if user_question:
    bills = ''
    if st.session_state.summary_history:
        for i, summary in enumerate(st.session_state.summary_history, 1):
            bills = bills + 'Bill ' + str(i) + '\n' + summary
        final = bills + '\n' + user_question
        print('ALL BILLS AND WHAT TO ASK: \n', final)
        answer = mistral_interaction(final)
        print('FINAL ANSWER: \n', answer)

        # Display and save the summary
        st.text(answer)