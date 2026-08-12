import streamlit as st
import os

# --- Stable Imports (No LangChain Memory required) ---
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration (Must match your notebook setup) ---
CHROMA_PATH = "ollama_mrag_db"
OLLAMA_MODEL = "nomic-embed-text" 
OLLAMA_LLM = "llava" # The MLLM for generation

# --- Prompt Template (For Grounding) ---
# We simplify the prompt since we manage history separately
RAG_PROMPT_TEMPLATE = """You are a Multimodal RAG assistant. Answer the user's question only based on the following context. 
If the context includes information extracted from images (look for summaries of charts or diagrams), integrate that visual information into your response. 
If the context does not contain the answer, state that you do not have enough information.
Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""

# --- 1. Load the RAG Index and Components ---

@st.cache_resource
def setup_rag_components():
    """Loads the vector store and initializes the LLM."""
    try:
        # Re-initialize the embedding model 
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
        # Initialize the Ollama MLLM
        llm = Ollama(model=OLLAMA_LLM, temperature=0)
        
        # We no longer return the memory object
        return llm, vectorstore
        
    except Exception as e:
        st.error(f"Error setting up RAG components. Is Ollama running? Error: {e}")
        return None, None

# --- 2. Streamlit UI Setup ---

st.set_page_config(page_title="Multimodal Ollama RAG Chatbot", layout="wide")
st.title("🤖 Local Multimodal RAG Chatbot (Ollama + Streamlit)")

llm, vectorstore = setup_rag_components() # Only two return values

# Initialize chat history (This is the new memory container)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me about the RL Intro PDF."}
    ]
if "history_list" not in st.session_state:
    st.session_state.history_list = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Handle User Input and RAG Logic (Functional Method without memory module) ---

if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        if llm and vectorstore:
            try:
                # 1. Retrieval
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(prompt)
                
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 2. Augmentation (Build the Prompt)
                # Format history directly from session state messages
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                                               for msg in st.session_state.history_list])
                
                final_prompt = RAG_PROMPT_TEMPLATE.format(
                    context=context,
                    chat_history=formatted_history,
                    question=prompt
                )

                # 3. Generation
                response = llm.invoke(final_prompt)
                
                # Update memory (manually append to history_list)
                st.session_state.history_list.append({"role": "user", "content": prompt})
                st.session_state.history_list.append({"role": "assistant", "content": response})
                
                # Format source information
                source_info = "\n\n**Sources:**\n"
                for doc in retrieved_docs:
                    doc_type = doc.metadata.get('type', 'text')
                    page = doc.metadata.get('page', 'N/A')
                    source_info += f"- Page {page}: ({doc_type} chunk)\n"
                
                final_response = response + source_info

            except Exception as e:
                final_response = f"An error occurred during generation: {e}"
        else:
            final_response = "RAG system not initialized. Check your Ollama server."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)
    st.session_state.messages.append({"role": "assistant", "content": final_response})