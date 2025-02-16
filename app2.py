import os
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



def load_api_key():
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)
        api_key = config_data.get('GROQ_API_KEY')
        if api_key is None:
            raise ValueError("API key not found in config.json")
        return api_key

grok_api_key = load_api_key()


def load_pdf_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    doc = loader.load()
    return doc


def load_word_document(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    doc = loader.load()
    return doc


def preprocess_text(doc):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    chunks = text_splitter.split_documents(doc)
    return chunks


def vec_database(chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_chain(vector_store):
    llm = ChatGroq(
        api_key=grok_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None   
    )
    retriever = vector_store.as_retriever()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    return conversational_chain



st.set_page_config(
    page_title="TextBit",
    page_icon="D:\Documents\LLM Project\Chatbot_4\OIG4.jpeg",
    layout="centered"
)

st.title(f"TextBit: Your AI Language Companion")

st.subheader("Your Personal AI Assistant 🤖")

st.sidebar.title("🔧 Options")

mode = st.sidebar.radio("Choose the mode of chat: ", ["💬 Chat with Text", "📰 Analyze Document"])

options = ["llama-3.3-70b-versatile","gemma2-9b-it", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

selected_model = st.sidebar.selectbox("Select the model: ", options)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if st.sidebar.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.session_state.conversation_chain = None
    st.session_state.document_processed = False


if mode == "💬 Chat with Text":
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    llm = ChatGroq(
        api_key=grok_api_key,
        model=selected_model,
        temperature=0,
        max_tokens=None
    )

    user_input = st.chat_input("Type your message here......")

    if user_input:
        st.chat_message("user").markdown(user_input)
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        full_conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            *st.session_state.chat_history
        ]

        response = llm.invoke(full_conversation + [{"role": "user", "content": user_input}])
        
        assistant_response = response.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)


elif mode == "📰 Analyze Document":
    file_option = ["PDF", "Word Document"]

    user_file_choice = st.sidebar.selectbox("Select the type of file", file_option)

    uploaded_file = None
    if user_file_choice == "PDF":
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    elif user_file_choice == "Word Document":
        uploaded_file = st.sidebar.file_uploader("Upload Word Document", type=["docx"])

    if uploaded_file and not st.session_state.get("vectorstore"):
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Loading and processing document..."):
            if user_file_choice == "PDF":
                doc = load_pdf_document(file_path)
            elif user_file_choice == "Word Document":
                doc = load_word_document(file_path)

            chunks = preprocess_text(doc)

            with st.spinner("Creating vector database..."):
                st.session_state.vectorstore = vec_database(chunks)

            with st.spinner("Connecting to Groq API..."):
                st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

        st.session_state.document_processed = True
        st.success("Document successfully processed!")


    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    user_input = st.chat_input("Ask about the document...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        if st.session_state.get("conversation_chain"):
            with st.chat_message("assistant"):
                response = st.session_state.conversation_chain({"question": user_input})
                assistant_response = response["answer"]

                st.markdown(assistant_response)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

