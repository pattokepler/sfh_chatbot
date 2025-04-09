import streamlit as st
import os
from groq import Groq
import random
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")


if not groq_api_key:
    raise ValueError("API_KEY is not set")

# Function to handle the vector embeddings and document loading
def vector_embedding():
    """
    This function handles the vector embeddings and document loading.
    It will load the PDFs from the "pdf" directory, split them into chunks,
    and create a vector store for search functionality.
    """
    if "vectors" not in st.session_state:
        # Use HuggingFace embeddings for document vectors
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load documents from the "pdf" directory
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        # Split documents into chunks of 1000 characters with 200 overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        
        # Create FAISS vector store for efficient search
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vectorization
        # Set vectorstore in session state
        st.session_state.vectorstore = st.session_state.vectors

# Initialize vectorstore if not available
if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
    try:
        print("Processing PDF file now...")  # Log when processing starts
        with st.spinner("Processing document..."):
            vector_embedding()  # Create the vectorstore
        st.success("ChatBot Initialized OK!")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
else:
    print("Vectorstore already exists, skipping PDF processing.")  # Log if vectorstore is already initialized

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('sfh.png')
    st.sidebar.image('sfh.png', width=150)
    
    # The title and greeting message of the Streamlit application
    st.title("SFH AI CHATBOT 🤖")
    st.write("Hello! I'm your friendly SFH AI chatbot. Let's start our conversation!")

    # Sidebar customization options
    # st.sidebar.title('Customization')

    # Allow users to toggle RAG on or off
    use_rag = st.sidebar.radio("Use PDF-based RAG?", ["No", "Yes"], index=0)
    
    system_prompt = st.sidebar.text_input("System prompt AI Agent:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'qwen-2.5-32b', 'gemma2-9b-it','deepseek-r1-distill-qwen-32b']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)
    
    st.sidebar.markdown("""
    <hr style="border: 1px solid #007bff;">
    <br>
    <span style="color: #007bff;">Society for Family Health (SFH) - Rwanda</span>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<small>http://www.sfhrwanda.org/ </small>", unsafe_allow_html=True)

    
    # Only show PDF uploader if RAG is selected
    if use_rag == "Yes":
        pdf_file = st.file_uploader("Upload a PDF for RAG", type="pdf", accept_multiple_files=True)
    
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question:")

    # session state variable for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    # If RAG is selected, process the uploaded PDF
    if use_rag == "Yes" and pdf_file:
        with st.spinner("Processing PDF documents..."):
            # Save uploaded PDFs to a directory
            pdf_dir = "./pdf"
            os.makedirs(pdf_dir, exist_ok=True)
            for uploaded_pdf in pdf_file:
                with open(os.path.join(pdf_dir, uploaded_pdf.name), "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
            # Reinitialize vector store after new PDFs are uploaded
            vector_embedding()

    # If the user has asked a question,
    if user_question:
        if use_rag == "Yes" and "vectorstore" in st.session_state:
            # Retrieve relevant documents using the vector store
            query_vector = st.session_state.embeddings.embed_query(user_question)
            results = st.session_state.vectorstore.similarity_search_by_vector(query_vector, k=3)  # Fetch top 3 similar documents
            
            # Combine the relevant documents with the user question for the prompt
            relevant_docs = "\n".join([doc.page_content for doc in results])
            augmented_question = f"Relevant information from documents:\n{relevant_docs}\n\nUser's question: {user_question}"

            # Construct a chat prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Create a conversation chain
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            # Generate the chatbot's response
            response = conversation.predict(human_input=augmented_question)
            message = {'human': user_question, 'AI': response}
            st.session_state.chat_history.append(message)
            st.write("Chatbot:", response)
        
        elif use_rag == "No":
            # If RAG is off, generate a response without document retrieval
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )
        
            # The chatbot's answer is generated by sending the full prompt to the Groq API.
            response = conversation.predict(human_input=user_question)
            message = {'human': user_question, 'AI': response}
            st.session_state.chat_history.append(message)
            st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
