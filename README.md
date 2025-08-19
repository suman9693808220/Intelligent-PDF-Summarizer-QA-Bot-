# Intelligent-PDF-Summarizer-QA-Bot-
PDF Chat App 📝📚
This project is a Streamlit-based application that allows users to upload a PDF document, ask questions about its content, and get context-aware answers. It uses LangChain, Google Generative AI, and a Retrieval-Augmented Generation (RAG) pipeline to achieve highly accurate and contextual responses.

Features 🚀
Upload PDFs: Users can upload any PDF document.
Chat Interface: Ask questions about the uploaded PDF content in a user-friendly chat interface.
RAG Pipeline: Utilizes LangChain for document splitting, embeddings creation, and retrieval.
Contextual Answers: Provides accurate answers based on the uploaded document.
Conversation History: Saves Q&A sessions with timestamps in a CSV file.
Streamlit Integration: A seamless and intuitive user experience.
Google Generative AI Integration: For generating embeddings and AI-based responses.
Tech Stack 🛠️
Python: Core programming language.
Streamlit: For creating the web application interface.
LangChain: For the RAG pipeline and LLM integration.
Google Generative AI: For embeddings and chat responses.
FAISS: For vector similarity search and retrieval.
How to Run the Project 🏃‍♂️
Prerequisites
Install Python 3.8+

Install the required libraries by running:

pip install -r requirements.txt
Create a .env file in the root directory with your Google Generative AI credentials:

GOOGLE_API_KEY=your_google_api_key
Steps to Run
Clone the repository:

[git clone https://github.com/your-repo-name/pdf-chat-app.git]
cd RAG-based_Chatbot
Start the Streamlit application:

streamlit run app.py
Open your browser and navigate to http://localhost:8501 to interact with the app.

File Structure 📂
pdf-chat-app/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── record.csv             # Logs conversations with timestamps (generated after interaction)
└── README.md              # Project documentation
Usage Instructions 🖐️
Upload a PDF document using the "Choose a PDF file" option.
Once the PDF is processed, type your question into the text input field.
View the AI-generated answer in real time.
Check the "Conversation History" for past interactions.
Example Q&A Session 💬
Timestamp	PDF Name	Question	Answer
10-01-2025 12:37:49	yolov9_paper.pdf	what is Auxiliary Supervision?	Auxiliary supervision is a common method that uses relevant meta-information to guide the feature maps produced by the intermediate layers, giving them the properties needed for target tasks. Examples include using segmentation loss or depth loss to improve the accuracy of object detectors. Deep supervision is the most common type of auxiliary supervision. It inserts additional prediction layers in the middle layers for training, such as the application of multi-layer decoders in transformer-based methods.
Streamlit Output🙌

