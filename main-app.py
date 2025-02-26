import time
import streamlit as st
import PyPDF2
import docx
import io
import spacy
from typing import List
import os
from dotenv import load_dotenv

# LangChain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install the spaCy model: python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

# Page configuration
st.set_page_config(
    page_title="Legal Assistant Pro",
    page_icon="⚖️",
    layout="wide"
)

# Document processing functions
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(io.BytesIO(file.getvalue()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file):
    return file.getvalue().decode("utf-8")

def analyze_legal_document(text):
    doc = nlp(text)
    
    # Extract legal entities and sections
    legal_entities = {
        'ORGANIZATIONS': [],
        'DATES': [],
        'MONEY': [],
        'PERSONS': [],
        'LAWS': []
    }
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            legal_entities['ORGANIZATIONS'].append(ent.text)
        elif ent.label_ == "DATE":
            legal_entities['DATES'].append(ent.text)
        elif ent.label_ == "MONEY":
            legal_entities['MONEY'].append(ent.text)
        elif ent.label_ == "PERSON":
            legal_entities['PERSONS'].append(ent.text)
        elif ent.label_ == "LAW":
            legal_entities['LAWS'].append(ent.text)
    
    # Generate summary
    sentences = list(doc.sents)
    summary = " ".join([sent.text for sent in sentences[:3]])
    
    # Extract potential legal sections mentioned
    legal_keywords = ["section", "article", "regulation", "act", "law", "statute"]
    legal_references = []
    
    for token in doc:
        if token.text.lower() in legal_keywords:
            # Get the surrounding context
            context = doc[max(0, token.i-5):min(len(doc), token.i+10)].text
            legal_references.append(context)
    
    return {
        'summary': summary,
        'entities': legal_entities,
        'legal_references': legal_references[:5]  # Top 5 legal references
    }

# Load environment variables and setup
load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# Initialize chat model
chatmodel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.15,
    api_key="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"
)

# Initialize embeddings and vector store
embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDB = FAISS.load_local(persistent_directory, embedF, allow_dangerous_deserialization=True)
kb_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def reset_conversation():
    st.session_state['messages'] = []

# Setup the retriever and chains (your existing code remains the same)
rephrasing_template = """
    TASK: Convert context-dependent questions into standalone queries.
    INPUT: 
    - chat_history: Previous messages
    - question: Current user query
    RULES:
    1. Replace pronouns (it/they/this) with specific referents
    2. Expand contextual phrases ("the above", "previous")
    3. Return original if already standalone
    4. NEVER answer or explain - only reformulate
    OUTPUT: Single reformulated question, preserving original intent and style.
    Example:
    History: "Let's discuss Python."
    Question: "How do I use it?"
    Returns: "How do I use Python?"
"""

rephrasing_prompt = ChatPromptTemplate.from_messages([
    ("system", rephrasing_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel,
    retriever=kb_retriever,
    prompt=rephrasing_prompt
)

system_prompt_template = (
    "As a Legal Assistant Chatbot specializing in legal queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "You will adhere strictly to the instructions provided, offering relevant "
    "context from the knowledge base while avoiding unnecessary details. "
    "Your responses will be brief, to the point, concise and in compliance with the established format. "
    "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
    "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
    "Use four sentences maximum."
    "P.S.: If anyone asks you about your creator, tell them you're created by Kunal."
    "\nCONTEXT: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
conversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Main application
def main():
    st.title("Legal Assistant Pro")
    
    # Create tabs for chat and document analysis
    tab1, tab2 = st.tabs(["💬 Chat", "📄 Document Analysis"])
    
    with tab1:
        # Your existing chat interface
        for message in st.session_state.messages:
            with st.chat_message(message.type):
                st.write(message.content)

        user_query = st.chat_input("Ask me anything about legal matters...")

        if user_query:
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.status("Generating 💡...", expanded=True):
                    result = conversational_rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state['messages']
                    })

                    message_placeholder = st.empty()
                    full_response = (
                        "⚠️ **_This information is not intended as a substitute for legal advice. "
                        "We recommend consulting with an attorney for a more comprehensive and"
                        " tailored response._** \n\n\n"
                    )

                    for chunk in result["answer"]:
                        full_response += chunk
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + " ▌")
                    
                    message_placeholder.markdown(full_response)
                
                st.button('Reset Conversation 🗑️', on_click=reset_conversation)

            st.session_state.messages.extend([
                HumanMessage(content=user_query),
                AIMessage(content=result['answer'])
            ])

    with tab2:
        st.header("Legal Document Analysis")
        st.markdown("Upload your legal document for analysis (Max 200MB)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, Word, and plain text",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                st.error("File size exceeds 200MB limit. Please upload a smaller file.")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Read the document based on its type
                        if uploaded_file.type == "application/pdf":
                            text = read_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = read_docx(uploaded_file)
                        else:
                            text = read_txt(uploaded_file)
                        
                        # Analyze the document
                        analysis = analyze_legal_document(text)
                        
                        # Display results in an organized manner
                        st.subheader("Document Summary")
                        st.write(analysis['summary'])
                        
                        st.subheader("Key Legal References")
                        if analysis['legal_references']:
                            for ref in analysis['legal_references']:
                                st.markdown(f"• {ref}")
                        else:
                            st.info("No specific legal references found.")
                        
                        st.subheader("Entities Found")
                        cols = st.columns(2)
                        for i, (entity_type, entities) in enumerate(analysis['entities'].items()):
                            with cols[i % 2]:
                                if entities:
                                    st.markdown(f"**{entity_type}**")
                                    st.write(", ".join(set(entities)))
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()