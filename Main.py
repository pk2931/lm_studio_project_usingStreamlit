import streamlit as st
from PIL import Image
# import pytesseract
from transformers import DonutProcessor, VisionEncoderDecoderModel
from openai import OpenAI
import re
import torch
# from google.cloud import vision

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Bill Manager",
    page_icon="📊",
)

st.sidebar.success("Add Bill/Check Summary")

# Initialize session state for storing history
if 'summary_history' not in st.session_state:
    st.session_state.summary_history = []

@st.cache_resource
def load_models():
    # Initialize models once using session state
    st.session_state.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    client = st.session_state.client
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if 'model' not in st.session_state:
        st.session_state.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(device)
        st.session_state.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    #google_vision = vision.ImageAnnotatorClient()
    
    return st.session_state.model, st.session_state.processor, client


model, processor, client = load_models()

# Function to format response text for readability
def format_response_text(text):
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n\n', text)
    text = re.sub(r'(\n)?(\s*)?([•\-*]|\d+\.)\s+', r'\n    \3 ', text)
    return text

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
        top_p=0.9,         # Limits token selection to a narrower subset
        max_tokens=300
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

def populate_lines(text):
    l = text.split('\n')
    s = ''
    for i in range(0, len(l), 2):
        try:
            s = s + l[i] + ' ' + l[i+1] + '\n'
        except:
            s = s + l[i] + '\n'
    return s
        

@st.cache_data
def processandsummarize(image):
    # Process the image
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    formatted_receipt = postprocess_receipt(processor.token2json(sequence))
    full_response = formatted_receipt + "\n \n Give me a strcutred summary of this bill including all the products, prices, etc in a neat way. Also dont mention it in the answer that you give just give me the result."
    
    full_response = mistral_interaction(full_response)
    full_response = full_response.replace(',', '.')
    #full_response = re.sub(r'[^A-Za-z0-9\s]', '', full_response)

    return full_response

# Add a checkbox for toggling history view
show_history = st.checkbox("Past Bill Summaries and Spending Insights", value=False)

if show_history:
    # Display the history of all summaries
    st.subheader("Summary History:")
    bills = ''
    if st.session_state.summary_history:
        for i, summary in enumerate(st.session_state.summary_history, 1):
            st.write(f"**Summary {i}:**")
            st.text(summary)
            bills = bills + 'Bill ' + str(i) + '\n' + summary
        question = 'On the basis of these bills, tell me how are my spendings. Also for more context these are restaurant bills, give a detailed explaination.'
        final = bills + '\n' + question
        print('ALL BILLS AND WHAT TO ASK: \n', final)
        answer = mistral_interaction(final)
        print('FINAL ANSWER: \n', answer)
        st.subheader('Spending patterns and tips to save! 🙂')
        st.text(answer)
    else:
        st.write("No summaries available.")
else:
    # File uploader widget for processing new bills
    uploaded_image = st.file_uploader("Upload an image of the receipt", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Resize the image before processing to speed up
        image = image.resize((1024, 1024))

        full_response = processandsummarize(image)
        full_response = populate_lines(full_response)

        # full_response = pytesseract.image_to_string(image)
        # full_response = full_response.replace(',', '.')
        # full_response = full_response + "\n \n Give me a strcutred summary of this bill including all the products, prices, etc in a neat way. Also dont mention it in the answer that you give just give me the result."
        # full_response = mistral_interaction(full_response)

        # response = client.text_detection(image=image)
        # texts = response.text_annotations
        # full_response = ''
        # for text in texts:
        #     full_response = full_response + text
        # full_response = full_response + "\n \n Give me a strcutred summary of this bill including all the products, prices, etc in a neat way. Also dont mention it in the answer that you give just give me the result."
        # full_response = mistral_interaction(full_response)

        # Display and save the summary
        st.subheader("Your Summary is Ready! 🙂")
        st.text(full_response)
        st.session_state.summary_history.append(full_response)