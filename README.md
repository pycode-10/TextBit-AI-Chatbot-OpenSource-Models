# TextBit-AI-Chatbot-OpenSource-Models
TextBit AI Chat Assistant

## Project Overview
The aim of the project is to create an advanced AI-powered chatbot that enables seamless interaction with multiple open-source language models. It provides users with the ability to chat with AI, analyze documents, and retrieve contextual responses efficiently. The integration of vector databases and conversational memory enhances user experience by maintaining context and improving response relevance.

## Technologies and Features

### Multiple AI Models
- Users can choose from several open-source models, allowing for a tailored chatbot experience based on their needs.
- Available models include:
  - `llama-3.3-70b-versatile`: A powerful large-scale model suitable for versatile applications.
  - `gemma2-9b-it`: Optimized for interactive and task-specific conversations.
  - `llama-3.1-8b-instant`: A lightweight and faster alternative for real-time responses.
  - `mixtral-8x7b-32768`: A mixture of experts model for handling diverse queries efficiently.

### Document Analysis
- Supports document uploads in both **PDF** and **Word** formats, making it easy for users to analyze and interact with their content.
- Extracts and processes text using advanced **natural language processing (NLP)** techniques.
- Allows users to perform intelligent querying over documents, enabling easy retrieval of specific information.
- Summarizes key points from documents, reducing the need for manual reading and interpretation.

### Vector Databases
- Uses **FAISS (Facebook AI Similarity Search)** for high-speed and scalable document search.
- Converts uploaded documents into vector embeddings, allowing for efficient and context-aware search queries.
- Provides fast and accurate results by retrieving the most relevant document sections based on user input.

### Conversational Chaining
- Maintains a **persistent conversation history**, ensuring that responses remain contextually aware throughout the interaction.
- Uses **LangChain’s conversation buffer memory** to store past exchanges and deliver coherent, logically connected responses.
- Enables dynamic topic-switching while retaining relevant information from previous discussions.

### Groq API Integration
- Leverages **Groq API** to fetch AI-generated responses from powerful open-source models.
- Provides users with access to **cutting-edge AI models** capable of answering complex queries and generating human-like responses.
- Ensures diverse response generation by connecting to a wide range of available AI models.

### Clear Chat History
- Users have the option to reset conversations at any time, ensuring a fresh start when needed.
- Helps prevent cluttered chat logs, improving usability and response accuracy over extended interactions.
- Useful for switching between different tasks or topics without interference from previous discussions.

## Cloud-Based Infrastructure for Low Latency
- The chatbot operates on **cloud-based infrastructure**, reducing computational overhead on local machines.
- Cloud deployment allows for **faster processing times**, leading to improved responsiveness and efficiency.
- AI computations and model interactions occur in a **distributed environment**, enhancing scalability and ensuring minimal downtime.
- By leveraging cloud resources, the chatbot is capable of handling multiple user queries simultaneously without performance degradation.

## Usage
Run the chatbot locally:
```bash
streamlit run app2.py
