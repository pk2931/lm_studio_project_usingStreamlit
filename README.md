# Bill Manager 📊

A streamlined application for managing restaurant bills and analyzing spending patterns using state-of-the-art OCR-free document understanding and advanced LLM-based insights.

## Features
- Upload restaurant bill images and extract structured summaries.
- Get spending insights based on historical bill data.
- Powered by **Donut (Document Understanding Transformer)** for OCR-free image processing.
- Advanced analysis and response generation using **LM Studio** with Mistral models.

---

## Folder Structure
lm_studio_project_usingStreamlit
│
├── Main.py             # Main script for processing bills and generating summaries.
│
├── Pages               # Folder for additional Streamlit pages.
│   └── User_Question.py # Module for user interaction or extended features.
│
├── Resources           # Supporting resources for the application.
│   └── Best            # Testing receipts
│
└── Readme.md           # Documentation for the project.

## Steps to Run the App
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lm_studio_project_usingStreamlit.git
cd lm_studio_project_usingStreamlit
```

### 2. Set Up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install the Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run Main.py
```

### 5. Access the App
Open your browser and go to http://localhost:8501.

Start uploading your bills and explore your spending patterns!
