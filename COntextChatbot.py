import os
import glob
import pickle
import json
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from google.cloud import storage
# Set the OpenAI API key
google_key = json.loads(st.secrets["google"]["key"])
user_apikey = st.secrets["openai"]["api_key"]
# Save it as a temporary file
with open("temp_key.json", "w") as f:
    json.dump(google_key, f)

# Paths for saving/loading processed data and conversation history
processed_texts_path = './temp/processed_texts.pkl'
faiss_index_path = 'faiss_index'
conversation_history_path = 'conversation_history.json'

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

def get_knowledge_base_path(selected_knowledge, bucket_name):
    """Downloads the knowledge base files from GCP and returns their local paths."""
    
    knowledge_base_dir = {
        "Child Growth and Development": r"KnowledgeBase/Child Growth and Development/",
        "Philosophical and Theoretical Perspectives in Education": r"KnowledgeBase/Philosophical and Theoretical Perspectives in Education/",
        "Pedagogical Studies": r"KnowledgeBase/Pedagogical Studies/",
        "Pedagogical Studies":r"KnowledgeBase/Pedagogical Studies/",
        "Curriculum Studies": r"KnowledgeBase/Curriculum Studies/",
        "Educational Assessment & Evaluation": r"KnowledgeBase/Educational Assessment & Evaluation/",
        "Safety and Security": r"KnowledgeBase/Safety and Security/",
        "Diversity, Equity and Inclusion - 1": r"KnowledgeBase/Diversity, Equity and Inclusion - 1/",
        "21st Century Skills - Holistic Education": r"KnowledgeBase/21st Century Skills - Holistic Education/",
        "Personal Professional Development": r"KnowledgeBase/Personal Professional Development/",
        "School Administration and Management": r"KnowledgeBase/School Administration and Management/",
        "Promoting Health and Wellness through Education": r"KnowledgeBase/Promoting Health and Wellness through Education/",
        "Guidance and Counselling": r"KnowledgeBase/Guidance and Counselling/",
        "Vocational Education & Training": r"KnowledgeBase/Vocational Education & Training/",
        "Educational Leadership & Management": r"KnowledgeBase/Educational Leadership & Management/",
        "Designing/Setting up a School": r"KnowledgeBase/Designing/",
        "Research Methodology": r"KnowledgeBase/Research Methodology/",
        "Diversity, Equity and Inclusion - 2": r"KnowledgeBase/Diversity, Equity and Inclusion - 2/",
        "Monitoring Implementation and Evaluation": r"KnowledgeBase/Monitoring Implementation and Evaluation/",
        "Public Private Partnership": r"KnowledgeBase/Public Private Partnership/"
    }
    
    base_path = knowledge_base_dir.get(selected_knowledge)
    
    if base_path:
        # Define GCP bucket paths
        processed_texts_blob = os.path.join(base_path, "processed_texts.pkl")
        embeddings_blob = os.path.join(base_path, "knowledge_base_embeddings.pkl")
        faiss_index_blob = os.path.join(base_path, "faiss_index.index/index.faiss")
        index_blob = os.path.join(base_path, "faiss_index.index/index.pkl")
        # Define local paths
        local_processed_texts = os.path.join("temp", "processed_texts.pkl")
        local_embeddings = os.path.join("temp", "knowledge_base_embeddings.pkl")
        local_faiss_index = "index.faiss"
        local_index = "index.pkl"

        # Create 'temp' directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Download files from GCP bucket to local directory
        download_blob(bucket_name, processed_texts_blob, local_processed_texts)
        download_blob(bucket_name, embeddings_blob, local_embeddings)
        download_blob(bucket_name, faiss_index_blob, local_faiss_index)
        download_blob(bucket_name, index_blob, local_index)

        return local_processed_texts, local_embeddings
    else:
        print("Knowledge base not found.")
        return None, None

# Save conversation history to JSON
def save_conversation_history(history):
    with open(conversation_history_path, 'w') as f:
        json.dump(history, f)

# Load conversation history from JSON
def load_conversation_history():
    if os.path.exists(conversation_history_path):
        with open(conversation_history_path, 'r') as f:
            return json.load(f)
    else:
        return []

# Save processed texts
def save_processed_data(texts):
    with open(processed_texts_path, 'wb') as f:
        pickle.dump(texts, f)

# Load processed texts
def load_processed_texts():
    with open(processed_texts_path, 'rb') as f:
        return pickle.load(f)

# Save FAISS index
def save_faiss_index(faiss_index):
    faiss_index.save_local(faiss_index_path)

# Load FAISS index
def load_faiss_index(embeddings):
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=pickle.TRUE)

# Check if processed data exists
if os.path.exists(processed_texts_path) and os.path.exists(faiss_index_path):
    texts = load_processed_texts()
    faiss_index = load_faiss_index(OpenAIEmbeddings(openai_api_key=user_apikey))
else:
    # Path to the "Dat" folder
    dat_folder_path = r'/content/drive/MyDrive/Data2'

    # Get all PDF file paths in the "Dat" folder and subfolders
    pdf_paths = glob.glob(os.path.join(dat_folder_path, '**', '*.pdf'), recursive=True)

    # Load PDF documents
    documents = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {path}: {e}")

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings for text chunks
    embeddings = OpenAIEmbeddings(openai_api_key=user_apikey)
    faiss_index = FAISS.from_documents(texts, embeddings)

    # Save the processed data
    save_processed_data(texts)
    save_faiss_index(faiss_index)

# Initialize the ChatOpenAI model for the LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=user_apikey)

# Define the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever()
)

# Function to answer questions using a conversation history stored in JSON
def answer_question(query):
    conversation_history = load_conversation_history()

    # Combine the conversation history with the current query
    context = " ".join([item["content"] for item in conversation_history])
    combined_prompt = f"{context}\nUser: {query}"

    response = qa_chain({"query": combined_prompt})

    # Update the conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response["result"]})
    print(response["result"])
    # Save the updated conversation history
    save_conversation_history(conversation_history)

    return response["result"]

# Streamlit Interface
# Streamlit Interface
# Streamlit Interface
st.title("ChatGPT-TTCA")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_key.json"
bucket_name = "amtstore"

# Input box for the OpenAI API key
user_api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Check if API key is provided
if user_api_key:
    # Set the OpenAI API key dynamically
    user_apikey = user_api_key

    # Paths for saving/loading processed data and conversation history
    processed_texts_path = './temp/processed_texts.pkl'
    faiss_index_path = 'faiss_index'
    conversation_history_path = 'conversation_history.json'

    # Knowledge base options
    knowledge_options = [
        "Child Growth and Development",
        "Philosophical and Theoretical Perspectives in Education",
        "Pedagogical Studies",
        "Curriculum Studies",
        "Educational Assessment & Evaluation",
        "Safety and Security",
        "Diversity, Equity and Inclusion - 1",
        "21st Century Skills - Holistic Education",
        "Personal Professional Development",
        "School Administration and Management",
        "Promoting Health and Wellness through Education",
        "Guidance and Counselling",
        "Vocational Education & Training",
        "Educational Leadership & Management",
        "Designing/Setting up a School",
        "Research Methodology",
        "Diversity, Equity and Inclusion - 2",
        "Monitoring Implementation and Evaluation",
        "Public Private Partnership"
    ]

    # Select a knowledge base
    selected_knowledge = st.selectbox("Select a knowledge base:", knowledge_options)
    knowledge_base_path, embeddings_path = get_knowledge_base_path(selected_knowledge, bucket_name)

    if not knowledge_base_path or not embeddings_path:
        st.error(f"Knowledge base files not found for {selected_knowledge}.")
    else:
        # Initialize FAISS index and embeddings
        if os.path.exists(processed_texts_path) and os.path.exists(faiss_index_path):
            texts = load_processed_texts()
            embeddings = OpenAIEmbeddings(openai_api_key=user_apikey)
            faiss_index = load_faiss_index(embeddings)
        else:
            # Placeholder for loading or processing documents
            st.error("Please upload the necessary files to proceed.")

        # Initialize the LLM and RAG chain
        if 'faiss_index' in locals():
            llm = ChatOpenAI(model="gpt-4", openai_api_key=user_apikey)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=faiss_index.as_retriever()
            )

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("Ask me anything:"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get answer from the chatbot
                answer = answer_question(prompt)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
else:
    st.warning("Please provide your OpenAI API key to start.")
