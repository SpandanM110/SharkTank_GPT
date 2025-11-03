"""
Shark Tank Analysis System - Streamlit Frontend
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import io
import base64
from langgraph_workflow import SharkTankAnalyzer
from config import LANGFUSE_ENABLED, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
import os

# Page configuration
st.set_page_config(
    page_title="Shark Tank AI Analyzer",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better visibility and readability
st.markdown("""
<style>
    /* Import Google Fonts for better readability */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        color: #1a202c;
        line-height: 1.7;
        background-color: #f8fafc;
    }
    
    /* Main header with gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.02em;
    }
    
    /* Chat message styling with high contrast */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 2px solid transparent;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #1976d2;
        margin-left: 3rem;
        color: #0d47a1;
        font-weight: 500;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 6px solid #7b1fa2;
        margin-right: 3rem;
        color: #4a148c;
        font-weight: 500;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 3px solid rgba(255,255,255,0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h4 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .metric-card p {
        color: #f7fafc;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Success probability with better visibility */
    .success-probability {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    /* Risk and warning indicators */
    .risk-warning {
        color: #d32f2f;
        font-weight: 700;
        font-size: 1.2rem;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 6px solid #d32f2f;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Enhanced recommendations */
    .recommendation {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid #4caf50;
        font-weight: 500;
        color: #1b5e20;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 4px solid #667eea;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Subsection headers */
    .subsection-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #cbd5e0;
    }
    
    /* Content text styling */
    .content-text {
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2d3748;
        margin: 1.5rem 0;
        font-weight: 400;
    }
    
    /* Lists styling */
    .custom-list {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    .custom-list li {
        margin: 0.8rem 0;
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Alert boxes with better visibility */
    .alert-info {
        background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%);
        color: #1a365d;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #3182ce;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        color: #22543d;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #38a169;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef5e7 0%, #fbd38d 100%);
        color: #744210;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #ed8936;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Data tables with better styling */
    .dataframe {
        font-size: 1rem;
        border-collapse: collapse;
        width: 100%;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-radius: 12px;
        overflow: hidden;
        background-color: white;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 15px 12px;
        text-align: left;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #e2e8f0;
        background-color: #f7fafc;
        font-size: 1rem;
        color: #2d3748;
    }
    
    .dataframe tr:nth-child(even) td {
        background-color: #edf2f7;
    }
    
    .dataframe tr:hover td {
        background-color: #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-right: 3px solid #e2e8f0;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 3px dashed #cbd5e0;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Markdown content styling */
    .markdown-text-container {
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2d3748;
        font-weight: 400;
    }
    
    /* Headers in markdown */
    .markdown-text-container h1 {
        color: #1a365d;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        border-bottom: 4px solid #667eea;
        padding-bottom: 0.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .markdown-text-container h2 {
        color: #2d3748;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #cbd5e0;
        padding-bottom: 0.5rem;
    }
    
    .markdown-text-container h3 {
        color: #4a5568;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
    }
    
    /* Lists in markdown */
    .markdown-text-container ul, .markdown-text-container ol {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    .markdown-text-container li {
        margin: 1rem 0;
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Code in markdown */
    .markdown-text-container code {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Blockquotes */
    .markdown-text-container blockquote {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-left: 6px solid #667eea;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
        font-style: italic;
        color: #4a5568;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 6px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 70px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        border-radius: 12px;
        gap: 1px;
        padding: 20px 30px;
        font-weight: 600;
        color: #4a5568;
        border: 3px solid transparent;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 3px solid #4c51bf;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        font-weight: 500;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        font-size: 1.1rem;
        font-weight: 500;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        line-height: 1.6;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SharkTankAnalyzer()
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

def display_chat_message(role, content, analysis_data=None):
    """Display a chat message with optional analysis data"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong style="font-size: 1.2rem; color: #0d47a1;">You:</strong><br>
            <div style="margin-top: 0.5rem; font-size: 1.1rem; line-height: 1.6;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong style="font-size: 1.2rem; color: #4a148c;">🦈 Shark Tank AI:</strong><br>
            <div style="margin-top: 0.5rem; font-size: 1.1rem; line-height: 1.6;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if analysis_data:
            display_analysis_results(analysis_data)

def display_analysis_results(analysis_data):
    """Display detailed analysis results with enhanced visibility"""
    if "success_prediction" in analysis_data:
        success_prob = analysis_data["success_prediction"]["probability"]
        confidence = analysis_data["success_prediction"]["confidence"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Success Probability</h4>
                <div class="success-probability">{success_prob:.1%}</div>
                <p style="font-size: 1.2rem; margin-top: 1rem;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Analysis Type</h4>
                <p style="font-size: 1.3rem; margin-top: 1rem;"><strong>{analysis_data.get('query_type', 'General Analysis').replace('_', ' ').title()}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display LLM Analysis with enhanced styling
    if "groq_analysis" in analysis_data:
        st.markdown('<h2 class="section-header">🤖 AI Analysis</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{analysis_data["groq_analysis"]["analysis"]}</div>', unsafe_allow_html=True)
    elif "AI Analysis" in analysis_data.get('final_report', ''):
        st.markdown('<h2 class="section-header">🤖 AI Analysis</h2>', unsafe_allow_html=True)
        # Extract AI Analysis section from final report
        report = analysis_data.get('final_report', '')
        ai_section_start = report.find("## AI Analysis")
        if ai_section_start != -1:
            ai_section_end = report.find("##", ai_section_start + 1)
            if ai_section_end == -1:
                ai_section = report[ai_section_start:]
            else:
                ai_section = report[ai_section_start:ai_section_end]
            # Remove the header and display the content
            ai_content = ai_section.replace("## AI Analysis", "").strip()
            st.markdown(f'<div class="content-text">{ai_content}</div>', unsafe_allow_html=True)
    
    # Display Shark-Specific Recommendations with enhanced styling
    if "shark_recommendations" in analysis_data and analysis_data["shark_recommendations"]:
        st.markdown('<h2 class="section-header">🦈 Shark-Specific Recommendations</h2>', unsafe_allow_html=True)
        for shark, rec in analysis_data["shark_recommendations"].items():
            st.markdown(f"""
            <div class="alert-info">
                <strong style="font-size: 1.4rem; color: #1a365d;">{shark}</strong><br>
                <div style="margin-top: 1rem; font-size: 1.1rem; line-height: 1.7;">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display recommendations with enhanced styling
    if "recommendations" in analysis_data and analysis_data["recommendations"]:
        st.markdown('<h2 class="section-header">💡 General Recommendations</h2>', unsafe_allow_html=True)
        for i, rec in enumerate(analysis_data["recommendations"], 1):
            st.markdown(f"""
            <div class="recommendation">
                <strong style="font-size: 1.2rem; color: #1b5e20;">{i}.</strong> 
                <span style="font-size: 1.1rem; line-height: 1.6;">{rec}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Display risk factors with enhanced styling
    if "success_prediction" in analysis_data and analysis_data["success_prediction"].get("risks"):
        st.markdown('<h2 class="section-header">⚠️ Risk Factors</h2>', unsafe_allow_html=True)
        for i, risk in enumerate(analysis_data["success_prediction"]["risks"], 1):
            st.markdown(f"""
            <div class="risk-warning">
                <strong style="font-size: 1.2rem; color: #d32f2f;">{i}.</strong> 
                <span style="font-size: 1.1rem; line-height: 1.6;">{risk}</span>
            </div>
            """, unsafe_allow_html=True)

def create_visualizations(analysis_data):
    """Create interactive visualizations"""
    if "visualizations" not in analysis_data:
        return
    
    viz_data = analysis_data["visualizations"]
    
    # Country comparison chart
    if "country_comparison" in viz_data:
        st.markdown("### 📊 Country Performance Comparison")
        country_data = viz_data["country_comparison"]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Success Rates by Country", "Average Ask Amounts by Country"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=country_data["countries"], y=country_data["success_rates"], 
                   name="Success Rate", marker_color="#1f77b4"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=country_data["countries"], y=country_data["avg_ask_amounts"], 
                   name="Avg Ask Amount", marker_color="#ff7f0e"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Industry analysis
    if "industry_analysis" in viz_data:
        st.markdown("### 🏭 Industry Performance Analysis")
        for country, industries in viz_data["industry_analysis"].items():
            if industries:
                st.markdown(f"#### {country.title()}")
                
                # Create industry success rate chart
                industry_df = pd.DataFrame([
                    {"Industry": industry, "Success Rate": stats["success_rate"] * 100, 
                     "Total Pitches": stats["total_pitches"]}
                    for industry, stats in list(industries.items())[:10]
                ])
                
                if not industry_df.empty:
                    fig = px.bar(industry_df, x="Industry", y="Success Rate", 
                               title=f"Success Rates by Industry - {country.title()}",
                               color="Success Rate", color_continuous_scale="Viridis")
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Shark profiles
    if "shark_profiles" in viz_data:
        st.markdown("### 🦈 Shark Investment Profiles")
        for country, sharks in viz_data["shark_profiles"].items():
            st.markdown(f"#### {country.title()} Sharks")
            
            shark_data = []
            for shark_name, profile in sharks.items():
                if profile["total_investments"] > 0:
                    shark_data.append({
                        "Shark": shark_name,
                        "Total Investments": profile["total_investments"],
                        "Total Amount": profile.get("total_amount", 0),
                        "Avg Investment": profile.get("avg_investment", 0)
                    })
            
            if shark_data:
                shark_df = pd.DataFrame(shark_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(shark_df, x="Shark", y="Total Investments",
                               title="Number of Investments by Shark")
                    fig.update_layout(xaxis_tickangle=-45, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(shark_df, x="Shark", y="Avg Investment",
                               title="Average Investment Amount by Shark")
                    fig.update_layout(xaxis_tickangle=-45, height=300)
                    st.plotly_chart(fig, use_container_width=True)

def process_file_upload(uploaded_file):
    """Process uploaded file and extract pitch information"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return f"Uploaded CSV file with {len(df)} rows and {len(df.columns)} columns. Please describe your pitch idea for analysis."
        elif uploaded_file.name.endswith(('.txt', '.md')):
            content = str(uploaded_file.read(), "utf-8")
            return f"Uploaded text file content:\n\n{content}\n\nPlease provide additional details about your pitch for analysis."
        else:
            return "Unsupported file format. Please upload a CSV or text file."
    except Exception as e:
        return f"Error processing file: {str(e)}"

def generate_report_download(analysis_data):
    """Generate downloadable report"""
    report_content = analysis_data.get("final_report", "No report available")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shark_tank_analysis_{timestamp}.md"
    
    # Create download button
    st.download_button(
        label="📥 Download Analysis Report",
        data=report_content,
        file_name=filename,
        mime="text/markdown"
    )

def display_langfuse_status():
    """Display Langfuse connection status"""
    if LANGFUSE_ENABLED and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        st.markdown("""
        <div class="alert-success">
            <strong style="font-size: 1.2rem; color: #22543d;">✅ Langfuse Connected</strong><br>
            <div style="margin-top: 0.5rem; font-size: 1rem;">
                Observability data is being collected and sent to your Langfuse dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add link to dashboard
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="{LANGFUSE_BASE_URL}" target="_blank" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.8rem 2rem;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 600;
                font-size: 1.1rem;
                display: inline-block;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            ">
                🌐 Open Langfuse Dashboard
            </a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-warning">
            <strong style="font-size: 1.2rem; color: #744210;">⚠️ Langfuse Not Configured</strong><br>
            <div style="margin-top: 0.5rem; font-size: 1rem;">
                Set up Langfuse API keys in your .env file to enable observability.
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_analysis_metrics():
    """Display analysis metrics from session state"""
    if not st.session_state.analysis_history:
        st.markdown("""
        <div class="alert-info">
            <strong>📊 No Analysis Data Yet</strong><br>
            Perform some analyses to see metrics here.
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<h3 class="subsection-header">📈 Analysis Metrics</h3>', unsafe_allow_html=True)
    
    # Calculate metrics
    total_analyses = len(st.session_state.analysis_history)
    
    # Success probability metrics
    success_probs = []
    query_types = []
    execution_times = []
    
    for analysis in st.session_state.analysis_history:
        if "success_prediction" in analysis["data"]:
            success_probs.append(analysis["data"]["success_prediction"]["probability"])
        if "query_type" in analysis["data"]:
            query_types.append(analysis["data"]["query_type"])
        if "timestamp" in analysis:
            # Estimate execution time (simplified)
            execution_times.append(2.5)  # Average estimated time
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        if success_probs:
            avg_success = sum(success_probs) / len(success_probs)
            st.metric("Avg Success Rate", f"{avg_success:.1%}")
        else:
            st.metric("Avg Success Rate", "N/A")
    
    with col3:
        if query_types:
            most_common = max(set(query_types), key=query_types.count)
            st.metric("Most Common Type", most_common.replace('_', ' ').title())
        else:
            st.metric("Most Common Type", "N/A")
    
    with col4:
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            st.metric("Avg Execution Time", f"{avg_time:.1f}s")
        else:
            st.metric("Avg Execution Time", "N/A")

def display_recent_traces():
    """Display recent analysis traces"""
    if not st.session_state.analysis_history:
        return
    
    st.markdown('<h3 class="subsection-header">🔍 Recent Analysis Traces</h3>', unsafe_allow_html=True)
    
    # Show last 5 analyses
    recent_analyses = st.session_state.analysis_history[-5:]
    
    for i, analysis in enumerate(reversed(recent_analyses)):
        with st.expander(f"Trace #{len(st.session_state.analysis_history) - i}: {analysis['query'][:50]}...", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query:**")
                st.text(analysis['query'])
                
                st.markdown("**Timestamp:**")
                st.text(analysis['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
            
            with col2:
                if "success_prediction" in analysis["data"]:
                    success_prob = analysis["data"]["success_prediction"]["probability"]
                    st.markdown("**Success Probability:**")
                    st.progress(success_prob)
                    st.text(f"{success_prob:.1%}")
                
                if "query_type" in analysis["data"]:
                    st.markdown("**Analysis Type:**")
                    st.text(analysis["data"]["query_type"].replace('_', ' ').title())

def display_performance_charts():
    """Display performance charts"""
    if not st.session_state.analysis_history:
        return
    
    st.markdown('<h3 class="subsection-header">📊 Performance Trends</h3>', unsafe_allow_html=True)
    
    # Prepare data for charts
    df_data = []
    for i, analysis in enumerate(st.session_state.analysis_history):
        if "success_prediction" in analysis["data"]:
            df_data.append({
                "Analysis #": i + 1,
                "Success Probability": analysis["data"]["success_prediction"]["probability"],
                "Query Type": analysis["data"].get("query_type", "unknown").replace('_', ' ').title(),
                "Timestamp": analysis["timestamp"]
            })
    
    if not df_data:
        st.info("No performance data available yet.")
        return
    
    df = pd.DataFrame(df_data)
    
    # Success probability trend
    fig1 = px.line(df, x="Analysis #", y="Success Probability", 
                   title="Success Probability Trend",
                   color_discrete_sequence=["#667eea"])
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Query type distribution
    if len(df) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            query_type_counts = df["Query Type"].value_counts()
            fig2 = px.pie(values=query_type_counts.values, names=query_type_counts.index,
                         title="Analysis Type Distribution")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.bar(df, x="Query Type", y="Success Probability",
                         title="Success Rate by Analysis Type")
            fig3.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)

def display_langfuse_dashboard():
    """Display Langfuse observability dashboard"""
    st.markdown('<h2 class="section-header">🔍 Langfuse Observability Dashboard</h2>', unsafe_allow_html=True)
    
    # Status section
    display_langfuse_status()
    
    st.markdown("---")
    
    # Metrics section
    display_analysis_metrics()
    
    st.markdown("---")
    
    # Performance charts
    display_performance_charts()
    
    st.markdown("---")
    
    # Recent traces
    display_recent_traces()
    
    # Additional info
    st.markdown("""
    <div class="alert-info">
        <strong>💡 Pro Tip:</strong><br>
        <div style="margin-top: 0.5rem;">
            For detailed traces, workflow execution details, and advanced analytics, 
            visit the full Langfuse dashboard using the link above. This embedded view 
            shows key metrics from your current session.
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function with enhanced styling and Langfuse dashboard"""
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🦈 Shark Tank AI Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <p style="font-size: 1.4rem; color: #4a5568; font-weight: 600; margin: 0;">
            Your AI-Powered Investment Analysis & Pitch Evaluation System
        </p>
        <p style="font-size: 1.1rem; color: #718096; margin: 0.8rem 0 0 0; line-height: 1.5;">
            Get intelligent insights, success predictions, and personalized recommendations for your Shark Tank pitch
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📊 Langfuse Dashboard", "📈 Quick Stats"])
    
    with tab1:
        # Original chat interface
        display_chat_interface()
    
    with tab2:
        # Langfuse observability dashboard
        display_langfuse_dashboard()
    
    with tab3:
        # Quick stats and history
        display_quick_stats()

def display_quick_stats():
    """Display quick stats and analysis history"""
    st.markdown('<h2 class="section-header">📈 Quick Statistics</h2>', unsafe_allow_html=True)
    
    if st.session_state.analysis_history:
        total_analyses = len(st.session_state.analysis_history)
        st.metric("Total Analyses", total_analyses)
        
        # Calculate average success probability
        success_probs = []
        for analysis in st.session_state.analysis_history:
            if "success_prediction" in analysis["data"]:
                success_probs.append(analysis["data"]["success_prediction"]["probability"])
        
        if success_probs:
            avg_success = sum(success_probs) / len(success_probs)
            st.metric("Avg Success Rate", f"{avg_success:.1%}")
        
        # Show recent analyses
        st.markdown('<h3 class="subsection-header">📚 Recent Analyses</h3>', unsafe_allow_html=True)
        for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
            with st.expander(f"Analysis #{len(st.session_state.analysis_history) - i}: {analysis['query'][:40]}...", expanded=False):
                st.text(f"Query: {analysis['query']}")
                st.text(f"Time: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if "success_prediction" in analysis["data"]:
                    prob = analysis["data"]["success_prediction"]["probability"]
                    st.progress(prob)
                    st.text(f"Success Probability: {prob:.1%}")
    else:
        st.info("No analyses performed yet. Start chatting to see statistics here!")

def display_chat_interface():
    """Display the main chat interface"""
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; color: white;">
            <h3 style="color: white; margin: 0; font-size: 1.4rem; font-weight: 700;">🎯 Quick Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample queries with better styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h3 style="color: #2d3748; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;">💡 Sample Queries</h3>
        </div>
        """, unsafe_allow_html=True)
        
        sample_queries = [
            "I want to pitch a food tech startup asking for $500k for 15% equity",
            "What are the most successful industries in Shark Tank?",
            "Compare investment patterns between US and India sharks",
            "Analyze my pitch: AI-powered fitness app, $1M ask, 20% equity",
            "What makes a successful pitch in the health industry?"
        ]
        
        for query in sample_queries:
            if st.button(f"💬 {query[:50]}...", key=f"sample_{hash(query)}"):
                st.session_state.user_input = query
        
        st.markdown("---")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.markdown("### 📚 Analysis History")
            for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
                if st.button(f"View: {analysis['query'][:30]}...", key=f"history_{i}"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": analysis['query']
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Here's your previous analysis:",
                        "analysis_data": analysis['data']
                    })
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Pitch Document")
        uploaded_file = st.file_uploader(
            "Upload your pitch deck, business plan, or data file",
            type=['csv', 'txt', 'md'],
            help="Upload files to get context for your analysis"
        )
        
        if uploaded_file:
            file_content = process_file_upload(uploaded_file)
            st.text_area("File Content Preview", file_content, height=100)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"],
                message.get("analysis_data")
            )
        
        # Chat input
        # Enhanced chat input styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                    padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="color: #2d3748; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 700; text-align: center;">
                💬 Chat with Shark Tank AI
            </h2>
            <p style="color: #4a5568; text-align: center; margin: 0 0 1.5rem 0; font-size: 1.1rem;">
                Ask me anything about your pitch or Shark Tank analysis...
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if "user_input" in st.session_state:
            user_input = st.session_state.user_input
            del st.session_state.user_input
        else:
            user_input = st.chat_input("Ask me about your Shark Tank pitch or investment analysis...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            display_chat_message("user", user_input)
            
            # Process with AI
            with st.spinner("🦈 Analyzing your pitch with AI agents..."):
                try:
                    analysis_result = st.session_state.analyzer.analyze(user_input)
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        "query": user_input,
                        "data": analysis_result,
                        "timestamp": datetime.now()
                    })
                    
                    # Display AI response
                    if "error" in analysis_result:
                        st.error(f"Analysis error: {analysis_result['error']}")
                    else:
                        display_chat_message("assistant", "Analysis complete! Here are the results:", analysis_result)
                        
                        # Show visualizations
                        create_visualizations(analysis_result)
                        
                        # Generate report download
                        if "final_report" in analysis_result:
                            st.markdown("---")
                            generate_report_download(analysis_result)
                            
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    display_chat_message("assistant", f"I encountered an error while analyzing your request: {str(e)}")
    
    with col2:
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        
        if st.session_state.analysis_history:
            total_analyses = len(st.session_state.analysis_history)
            st.metric("Total Analyses", total_analyses)
            
            # Calculate average success probability
            success_probs = []
            for analysis in st.session_state.analysis_history:
                if "success_prediction" in analysis["data"]:
                    success_probs.append(analysis["data"]["success_prediction"]["probability"])
            
            if success_probs:
                avg_success = sum(success_probs) / len(success_probs)
                st.metric("Avg Success Rate", f"{avg_success:.1%}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Export data button
        if st.session_state.analysis_history:
            if st.button("📊 Export All Data"):
                export_data = {
                    "analyses": st.session_state.analysis_history,
                    "export_timestamp": datetime.now().isoformat()
                }
                
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="Download Analysis Data",
                    data=json_str,
                    file_name=f"shark_tank_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
