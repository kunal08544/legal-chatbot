import time
import streamlit as st
import spacy
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# LangChain dependencies
# Try alternative import paths
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
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

# Add new function for legal insights
def create_advanced_legal_insights_tab():
    st.header("🏛️ Comprehensive Legal Insights Platform")
    
    # Enhanced sections with advanced legal technology features
    sections = [
        "Judicial Performance Analytics",
        "Constitutional Rights Navigator",
        "Legal Risk Assessment", 
        "Landmark Judgments Analyzer",
        "Legal Trend Forecaster"
    ]
    
    selected_section = st.selectbox("Choose Legal Insight Domain", sections)
    
    if selected_section == "Judicial Performance Analytics":
        st.subheader("Indian Judiciary Performance Dashboard")
        
        # Supreme Court and High Court Performance Metrics
        court_performance = {
            'Court': ['Supreme Court', 'Delhi High Court', 'Bombay High Court', 'Calcutta High Court'],
            'Pending Cases': [70000, 45000, 60000, 55000],
            'Disposal Rate (%)': [65, 72, 68, 70],
            'Average Case Duration (Months)': [36, 24, 30, 28]
        }
        df_performance = pd.DataFrame(court_performance)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pending = px.bar(
                df_performance, 
                x='Court', 
                y='Pending Cases', 
                title='Pending Cases by Court',
                color='Disposal Rate (%)',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_pending)
        
        with col2:
            fig_duration = px.bar(
                df_performance, 
                x='Court', 
                y='Average Case Duration (Months)', 
                title='Case Resolution Duration',
                color='Disposal Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_duration)
        
        # Case Complexity Heatmap
        st.subheader("Case Complexity Analysis")
        complexity_data = {
            'Case Type': ['Criminal', 'Civil', 'Corporate', 'Constitutional', 'Labor'],
            'Complexity Score': [8.5, 7.2, 6.8, 9.1, 6.5],
            'Resolution Time (Months)': [48, 36, 24, 60, 18],
            'Success Probability (%)': [45, 55, 60, 40, 65]
        }
        df_complexity = pd.DataFrame(complexity_data)
        
        fig_complexity = px.scatter(
            df_complexity, 
            x='Complexity Score', 
            y='Resolution Time (Months)', 
            size='Success Probability (%)',
            color='Case Type',
            hover_name='Case Type',
            title='Legal Case Complexity Matrix',
            labels={'Complexity Score': 'Complexity Index', 'Resolution Time (Months)': 'Average Resolution Duration'}
        )
        st.plotly_chart(fig_complexity)
    
    elif selected_section == "Constitutional Rights Navigator":
        st.subheader("Indian Constitutional Rights Explorer")
        
        rights_categories = {
            "Fundamental Rights": [
                "Right to Equality (Article 14-18)",
                "Right to Freedom (Article 19-22)",
                "Right against Exploitation (Article 23-24)",
                "Right to Freedom of Religion (Article 25-28)",
                "Cultural and Educational Rights (Article 29-30)"
            ],
            "Directive Principles": [
                "Social Justice (Article 38)",
                "Right to Work (Article 41)",
                "Right to Education (Article 45)",
                "Protection of Environment (Article 48A)"
            ]
        }
        
        for category, rights in rights_categories.items():
            st.markdown(f"**{category}:**")
            for right in rights:
                st.markdown(f"• {right}")
    
    elif selected_section == "Legal Risk Assessment":
        st.subheader("Legal Risk Probability Calculator")
        
        risk_factors = {
            "Evidence Strength": st.slider("Evidence Strength", 0, 10, 5),
            "Legal Precedence": st.slider("Legal Precedence", 0, 10, 5),
            "Litigation Complexity": st.slider("Litigation Complexity", 0, 10, 5)
        }
        
        total_score = sum(risk_factors.values())
        risk_probability = min((total_score / 30) * 100, 100)
        
        risk_color = "green" if risk_probability < 30 else "yellow" if risk_probability < 60 else "red"
        
        st.markdown(f"""
        ### Risk Assessment Result
        **Estimated Legal Risk Probability:** 
        <span style='color:{risk_color}'>{risk_probability:.2f}%</span>
        """, unsafe_allow_html=True)
    
    elif selected_section == "Landmark Judgments Analyzer":
        st.subheader("Significant Indian Legal Precedents")
        
        landmark_cases = {
            "Kesavananda Bharati Case": "Defined Basic Structure Doctrine of Constitution",
            "Maneka Gandhi Case": "Expanded Scope of Personal Liberty",
            "Shah Bano Case": "Discussed Alimony Rights",
            "Right to Privacy Case": "Declared Privacy as Fundamental Right"
        }
        
        selected_case = st.selectbox("Select Landmark Case", list(landmark_cases.keys()))
        
        if selected_case:
            st.write(f"**Summary:** {landmark_cases[selected_case]}")
            
            # Additional details can be added here
            st.markdown("### Key Implications")
            st.markdown("- Significant judicial interpretation")
            st.markdown("- Established important legal principles")
    
    elif selected_section == "Legal Trend Forecaster":
        st.subheader("Legal Trends and Predictive Analysis")
        
        trend_areas = [
            "Corporate Law",
            "Intellectual Property",
            "Technology and Cyber Laws",
            "Environmental Regulations"
        ]
        
        selected_trend = st.selectbox("Select Trend Area", trend_areas)
        
        trend_data = {
            "Corporate Law": {
                "Growth Potential": 75,
                "Key Drivers": ["Digitalization", "Global Compliance"],
                "Emerging Focus": "ESG Compliance"
            },
            "Intellectual Property": {
                "Growth Potential": 65,
                "Key Drivers": ["Tech Innovation", "Startup Ecosystem"],
                "Emerging Focus": "AI and Patent Laws"
            },
            "Technology and Cyber Laws": {
                "Growth Potential": 85,
                "Key Drivers": ["Digital Transformation", "Data Protection"],
                "Emerging Focus": "AI Regulation"
            },
            "Environmental Regulations": {
                "Growth Potential": 60,
                "Key Drivers": ["Climate Change", "Sustainability"],
                "Emerging Focus": "Carbon Neutrality"
            }
        }
        
        if selected_trend:
            trend_info = trend_data[selected_trend]
            st.metric("Growth Potential", f"{trend_info['Growth Potential']}%")
            
            st.markdown("### Key Drivers")
            for driver in trend_info['Key Drivers']:
                st.markdown(f"• {driver}")
            
            st.markdown(f"### Emerging Focus: {trend_info['Emerging Focus']}")

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
    
    # Create tabs for chat and legal insights
    tab1, tab2 = st.tabs(["💬 Legal Chat", "🏛️ Legal Insights"])
    
    with tab1:
        # Chat interface
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
        # Legal Insights Tab
        create_advanced_legal_insights_tab()

if __name__ == "__main__":
    main()