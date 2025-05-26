# Medical Chatbot using Fine-tuned T5 Transformer

This project is a complete medical chatbot built using a fine-tuned T5 transformer model. The chatbot assists users with medical queries by generating responses based on our custom-trained FLAN-T5 model. The project includes a FastAPI backend and a modern HTML frontend interface.

## Project Overview

This is an updated and improved version of our previous medical chatbot project. We initially built a chatbot using NLTK and spaCy libraries (available at: https://github.com/arbaaz04/AI-Project-NextBot), but switched to this transformer-based approach as the previous implementation was outdated and not scalable for modern applications.

### Backend
- **Model**: Fine-tuned FLAN-T5 base model specifically trained on medical datasets using Google Colab
- **API**: FastAPI backend with `/chat` endpoint for handling user queries
- **Response Filtering**: Intelligent filtering system that removes unwanted phrases like "regards", "wish you a very", "i have read your question", and "thanks for"
- **Device Support**: Supports both CPU and MPS (Metal Performance Shaders) for efficient inference on macOS

### Frontend
- **Modern Interface**: Clean, responsive HTML interface with medical-themed design
- **Real-time Chat**: Interactive chat interface that communicates with the backend API
- **User Experience**: Professional medical assistant styling with typing indicators and smooth animations

### Model Training
- **Platform**: Model was fine-tuned using Google Colab with GPU acceleration
- **Base Model**: FLAN-T5 base model fine-tuned for medical conversations
- **Training Details**: Comprehensive training documentation available in separate documentation

## Features
- ✅ Fine-tuned FLAN-T5 transformer for medical chatbot functionality
- ✅ FastAPI backend with robust `/chat` endpoint
- ✅ Complete interactive HTML frontend
- ✅ Advanced response filtering for cleaner outputs
- ✅ Modern, responsive web interface
- ✅ Real-time chat functionality

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI-Term-Project-
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

   Keep in mind that it will download the 1GB model file from Google Drive.

5. Access the application at `http://127.0.0.1:8000`

## How to train the model

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI-Term-Project-
   ```
2. Install dependencies:
   ```bash
   pip install -r training_requirements.txt
   ```
3. Run the training_script.py file. It will download the preprocesed data from Google Drive and start training.

## Project Structure
```
AI-project-new/
├── app/
│   └── main.py                 # FastAPI backend with T5 model
├── static/
│   └── index.html             # Frontend interface
├── flan-t5-base-medical-chatbot-finetuned/  # Fine-tuned model files
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Technical Details

### Model Information
- **Base Model**: Google FLAN-T5 Base
- **Fine-tuning**: Custom medical dataset training via Google Colab
- **Tokenizer**: T5Tokenizer with SentencePiece
- **Generation**: Sampling-based generation with temperature=0.9

### API Endpoints
- `GET /`: Serves the frontend interface
- `POST /chat`: Accepts user messages and returns filtered chatbot responses

### Response Filtering
The chatbot includes intelligent filtering to remove common AI-generated phrases:
- Sentences containing "regards"
- Sentences containing "wish you a very"
- Sentences containing "i have read your question"
- Sentences containing "thanks for"

## Previous Work
This project builds upon our earlier chatbot implementation using NLTK and spaCy libraries. The previous version can be found at: https://github.com/arbaaz04/AI-Project-NextBot

We transitioned to this transformer-based approach because:
- Better contextual understanding
- More natural response generation
- Improved scalability and performance
- Modern architecture suitable for production use

## Project Purpose
This chatbot is developed for educational and research purposes as part of our AI project coursework. It demonstrates the implementation of modern transformer models for domain-specific applications.