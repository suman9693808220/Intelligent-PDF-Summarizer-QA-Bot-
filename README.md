Intelligent PDF Summarizer & QA Bot 📄🤖

This project is a Streamlit-based AI application that allows users to upload PDFs, process YouTube videos, or fetch website content, and then summarize and ask questions about the material. It leverages LangChain, Google Generative AI (Gemini), and a Retrieval-Augmented Generation (RAG) pipeline to deliver context-aware answers and multi-language summaries.

Features 🚀

Multi-Source Input

Upload PDFs

Provide Website URLs

Process YouTube videos (with transcript extraction)

Summarization

Generate concise summaries in English and Hindi

Download summaries as text files

Question Answering (RAG)

Ask questions about PDFs, URLs, or YouTube videos

Context-aware answers using FAISS vector search + Gemini

Conversation History

Saves Q&A sessions with timestamps, source type, and answers

Export history to CSV for record-keeping

Streamlit UI

Clean and interactive tab-based interface

Content upload, summarization, Q&A, and history tracking

Tech Stack 🛠️

Python – Core programming language

Streamlit – Web application interface

LangChain – RAG pipeline (document loaders, text splitters, embeddings, retrieval)

Google Generative AI (Gemini) – Embeddings & AI-based responses

FAISS – Vector similarity search & retrieval

PyPDF2 / PyMuPDF – PDF text extraction

YouTubeTranscriptApi – YouTube transcript extraction

How to Run the Project 🏃‍♂️
Prerequisites

Install Python 3.8+

Install dependencies:

pip install -r requirements.txt

Environment Setup

Create a .env file in the root directory and add your Google Gemini API key:

GOOGLE_API_KEY=your_google_api_key_here

Steps to Run

Clone the repository:

git clone https://github.com/your-repo-name/Intelligent-PDF-Summarizer-QA-Bot.git
cd Intelligent-PDF-Summarizer-QA-Bot


Start the Streamlit application:

streamlit run app.py


Open your browser and navigate to http://localhost:8501
 to interact with the app.

File Structure 📂
Intelligent-PDF-Summarizer-QA-Bot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── conversation_history.csv # Logs Q&A sessions (auto-generated)
├── .env                   # Environment file (Google API key)
└── README.md              # Project documentation

Usage Instructions 🖐️

Upload Content

Choose a PDF, enter a URL, or provide a YouTube link.

Process the content to build the knowledge base.

Summarize

Generate summaries in English or Hindi.

Download them as .txt files.

Ask Questions

Enter questions in the chat box.

Get accurate, context-aware answers.

View History

Browse past Q&A interactions with timestamps.

Example Q&A Session 💬
Timestamp	Source Type	Question	Answer
2025-01-18 15:22:31	PDF (AI_paper.pdf)	What is Retrieval-Augmented Generation?	Retrieval-Augmented Generation (RAG) is an AI technique that combines retrieval of relevant documents with generative language models to provide more accurate, context-aware responses.
Streamlit Output 🙌

📄 PDF Upload & Processing → Summaries → Q&A → Conversation History

![401876937-90b2d1bb-409b-4354-9fff-181312af22f9](https://github.com/user-attachments/assets/51c9293f-5ba9-4f00-967b-51c242d7a514)
![401876868-049addd1-cb29-4025-9666-fc2d09c1a44e](https://github.com/user-attachments/assets/959409a7-a743-4143-b33b-8bdc3e4d00d4)

✨ With this bot, you can transform static documents and videos into interactive, intelligent knowledge assistants!

