# Digital Legal Assistant for Legal Awareness & Document Design

An AI-powered **Digital Legal Assistant** built using **Natural Language Processing (NLP)** and **Sentence-BERT** to provide accurate legal information, multilingual support, voice-based interaction, document analysis, and automated legal document generation through an interactive **Streamlit** web interface.

---

## Project Overview
Accessing legal information is often difficult due to complex language, lack of awareness, and limited accessibility. This project addresses these challenges by developing an intelligent digital assistant that retrieves **contextually relevant legal answers** using **semantic similarity** instead of traditional keyword matching.

The assistant supports **text, voice, and document-based queries**, provides **multilingual interaction**, and enables users to **generate legal documents** by filling predefined templates dynamically.

---

## ✨ Key Features
- **Semantic Legal Question Answering** using Sentence-BERT  
- **Multilingual Support** (English, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali)  
- **Voice Input** using Speech-to-Text  
- **Document Upload & Analysis** (PDF, DOCX, TXT, Images via OCR)  
- **Context-Aware Answer Retrieval** using cosine similarity  
- **Legal Document Template Generator** (Auto-filled Word documents)  
- **Interactive Web App** built with Streamlit  

---

## Methodology
- Legal QA datasets collected from **Hugging Face (Legal-FAQ, Lawyer GPT India)**
- Text preprocessing and normalization
- Fine-tuned **Sentence-BERT (paraphrase-MiniLM-L6-v2)** for legal domain
- Precomputed embeddings stored for fast retrieval
- User queries matched using **cosine similarity**
- Best matching legal answer retrieved based on semantic meaning

---

## Model Performance
- **Evaluation Metric:** Cosine Similarity  
- **Average Similarity Score:** **0.9457**  
- Demonstrated strong understanding of paraphrased and multilingual legal queries  
- Outperformed traditional TF-IDF + Logistic Regression models in flexibility and contextual accuracy  

---

## Technologies Used
### Programming & Frameworks
- Python 3.8+
- Streamlit

### NLP & ML
- Sentence-Transformers (Sentence-BERT)
- PyTorch
- Cosine Similarity (Semantic Retrieval)

### Data & Utilities
- Pandas, NumPy
- Googletrans (Multilingual Translation)
- SpeechRecognition (Voice Input)
- Pytesseract (OCR)
- pdfplumber (PDF Text Extraction)
- python-docx (Document Automation)

---

## System Architecture
1. User provides input via text, voice, or document upload  
2. Input is translated to English (if required)  
3. Text is cleaned and converted into embeddings  
4. Semantic similarity is computed against stored embeddings  
5. Best-matched legal answer is retrieved  
6. Response is translated back to the selected language  
7. Optional: Legal document template is filled and generated  

---

## How to Run the Project
```bash
# Clone the repository
git clone https://github.com/your-username/digital-legal-assistant.git

# Navigate to the project directory
cd digital-legal-assistant

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run final.py
