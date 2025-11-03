"""
Demo script showing the new Langfuse Dashboard in Streamlit
"""
import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px

def demo_langfuse_dashboard():
    """Demo the Langfuse dashboard features"""
    
    st.title("🔍 Langfuse Dashboard Demo")
    st.markdown("This shows what you'll see in the new Langfuse Dashboard tab")
    
    # Simulate some analysis history
    demo_history = [
        {
            "query": "I want to pitch a food tech startup asking for $500k for 15% equity",
            "data": {
                "success_prediction": {"probability": 0.75},
                "query_type": "pitch_evaluation"
            },
            "timestamp": datetime.now()
        },
        {
            "query": "What are the most successful industries in Shark Tank?",
            "data": {
                "success_prediction": {"probability": 0.85},
                "query_type": "industry_analysis"
            },
            "timestamp": datetime.now()
        },
        {
            "query": "Compare investment patterns between US and India sharks",
            "data": {
                "success_prediction": {"probability": 0.65},
                "query_type": "country_comparison"
            },
            "timestamp": datetime.now()
        }
    ]
    
    # Status section
    st.markdown("### ✅ Langfuse Connection Status")
    st.success("✅ Langfuse Connected - Observability data is being collected!")
    
    # Link to dashboard
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <a href="https://cloud.langfuse.com" target="_blank" style="
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
    
    st.markdown("---")
    
    # Metrics section
    st.markdown("### 📈 Analysis Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(demo_history))
    
    with col2:
        avg_success = sum([h["data"]["success_prediction"]["probability"] for h in demo_history]) / len(demo_history)
        st.metric("Avg Success Rate", f"{avg_success:.1%}")
    
    with col3:
        st.metric("Most Common Type", "Pitch Evaluation")
    
    with col4:
        st.metric("Avg Execution Time", "2.5s")
    
    st.markdown("---")
    
    # Performance charts
    st.markdown("### 📊 Performance Trends")
    
    # Create demo data
    df_data = []
    for i, analysis in enumerate(demo_history):
        df_data.append({
            "Analysis #": i + 1,
            "Success Probability": analysis["data"]["success_prediction"]["probability"],
            "Query Type": analysis["data"]["query_type"].replace('_', ' ').title()
        })
    
    df = pd.DataFrame(df_data)
    
    # Success probability trend
    fig1 = px.line(df, x="Analysis #", y="Success Probability", 
                   title="Success Probability Trend",
                   color_discrete_sequence=["#667eea"])
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Query type distribution
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
    
    st.markdown("---")
    
    # Recent traces
    st.markdown("### 🔍 Recent Analysis Traces")
    
    for i, analysis in enumerate(reversed(demo_history)):
        with st.expander(f"Trace #{len(demo_history) - i}: {analysis['query'][:50]}...", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query:**")
                st.text(analysis['query'])
                
                st.markdown("**Timestamp:**")
                st.text(analysis['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
            
            with col2:
                success_prob = analysis["data"]["success_prediction"]["probability"]
                st.markdown("**Success Probability:**")
                st.progress(success_prob)
                st.text(f"{success_prob:.1%}")
                
                st.markdown("**Analysis Type:**")
                st.text(analysis["data"]["query_type"].replace('_', ' ').title())
    
    # Additional info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%);
                color: #1a365d;
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 6px solid #3182ce;
                margin: 1.5rem 0;
                font-weight: 600;
                font-size: 1.1rem;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <strong>💡 Pro Tip:</strong><br>
        <div style="margin-top: 0.5rem;">
            For detailed traces, workflow execution details, and advanced analytics, 
            visit the full Langfuse dashboard using the link above. This embedded view 
            shows key metrics from your current session.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    demo_langfuse_dashboard()

