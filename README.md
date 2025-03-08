AI-Powered PDF Analyzer with Ollama

Overview

This project is a local AI-powered PDF analysis tool that allows users to upload a PDF file and interactively ask questions about its contents. Using Ollama, a local machine AI model, the system processes the document, extracts key information, and enables users to query the file in a conversational manner.

Features

📄 PDF Upload & Parsing: Extracts text from PDFs for easy analysis.

🤖 AI-Powered Q&A: Uses Ollama's LLM to answer user queries based on document content.

🧠 Context-Aware Responses: Provides relevant, document-specific answers.

🔒 Offline & Secure: Runs entirely on the local machine, ensuring data privacy.

📂 Multi-Document Support: Analyze multiple PDFs for better insights.

Installation

Prerequisites

Python 3.8+

Pip

Ollama installed locally

Required Python libraries (see requirements.txt)

Setup

Clone the Repository

git clone https://github.com/yourusername/pdf-ai-analyzer.git
cd pdf-ai-analyzer

Install Dependencies

pip install -r requirements.txt

Run the Application

python start-1.py

Usage

Upload a PDF through the web UI.

Ask any question related to the document.

The AI processes the text and provides answers instantly.

Tech Stack

Backend: Python

AI Model: Ollama (local LLM)

PDF Processing: PyMuPDF (fitz) / PDFMiner

Frontend: Streamlit

Contributing

Feel free to fork the repo, submit issues, or create pull requests!

Contact

For any inquiries, reach out to youssef.waeldd@gmail.com or open an issue on GitHub.
