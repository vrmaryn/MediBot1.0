
#-----------------------------------------------------------------------------------------------
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st 
from langchain_community.llms import Ollama 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
import os

# Setup Wikipedia tool
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

# Setup arXiv tool 
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

 # Setup LLM
def load_llm():
   return Ollama(model="tinyllama")

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Modified prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a helpful medical assistant. Follow these instructions carefully:

1. For casual messages (no medical questions):
   - If the user says only "hi", "hello", "hey", etc.: Respond with "Hello! How can I help you with your medical questions today?"
   - If the user says "thank you", "thanks", etc.: Respond with "You're welcome! Let me know if you have any other questions."
   - If the user says "bye", "goodbye", etc.: Respond with "Goodbye! Take care and feel free to return if you have more questions."

2. For medical questions:
   - First, use the information in the context from our medical database if available.
   - If the question isn't covered in the context, I'll try to provide information from Wikipedia or arXiv.
   - If using external sources, clearly state: "This information comes from [source]" and summarize key points.
   - Always be clear, concise, and accurate.

Context: {context}
External Information: {external_info}
Question: {question}

Answer:
"""

# Create the prompt template object
prompt = PromptTemplate(
    input_variables=["context", "external_info", "question"],
    template=CUSTOM_PROMPT_TEMPLATE
)

def get_external_info(query):
    """Get information from external sources when local database doesn't have an answer"""
    wiki_info = ""
    arxiv_info = ""
    
    try:
        wiki_info = f"WIKIPEDIA: {wikipedia_tool.run(query)}"
    except Exception as e:
        wiki_info = "No relevant Wikipedia information found."
    
    try:
        arxiv_info = f"ARXIV: {arxiv_tool.run(query)}"
    except Exception as e:
        arxiv_info = "No relevant arXiv papers found."
    
    return f"{wiki_info}\n\n{arxiv_info}"

# Modified ConversationalRetrievalChain
def get_response(user_query):
    # First try to get response from our knowledge base
    docs = db.similarity_search(user_query, k=5)
    
    # If no relevant docs or if confidence is low
    if not docs or docs[0].metadata.get('score', 0) < 0.7:  # Adjust threshold as needed
        external_info = get_external_info(user_query)
    else:
        external_info = "No external information needed."
    
    # Get the context from the retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Process with the LLM
    llm_chain = LLMChain(llm=load_llm(), prompt=prompt)
    response = llm_chain.run(context=context, external_info=external_info, question=user_query)
    
    return response

# Streamlit UI
st.set_page_config(page_title="MediBot - Chat Mode", page_icon="💬")
st.title("💬 MediBot - Medical Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        answer = get_response(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("MediBot", answer))
        st.success("Answer:")
        st.write(answer)

# Show chat history
# Create tabs for chat and medical info
tab1, tab2 = st.tabs(["Chat", "Medical Info"])

with tab1:
    # Show chat history
    st.markdown("### 🗨️ Chat History")
    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")

with tab2:
    st.header("Common Medical Terms")

    # Create columns for organizing content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cardiovascular")
        st.markdown("- Hypertension (High Blood Pressure)")
        st.markdown("- Atherosclerosis (Hardening of arteries)")
        st.markdown("- Myocardial Infarction (Heart Attack)")

    with col2:
        st.subheader("Respiratory")
        st.markdown("- Asthma (Chronic lung condition)")
        st.markdown("- Pneumonia (Lung infection)")
        st.markdown("- COPD (Chronic Obstructive Pulmonary Disease)")

    st.divider()

    # Sample medical data (for demonstration)
    st.subheader("Medical Knowledge Base Statistics")

    import pandas as pd
    data = {
        'Source': ['Medical PDFs', 'Wikipedia', 'arXiv'],
        'Articles': [150, 5000, 2500],
        'Last Updated': ['2025-05-01', '2025-05-10', '2025-05-08']
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
