"""
LangGraph-based Multi-Agent Shark Tank Analysis System
"""
import pandas as pd
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langfuse import Langfuse
import time
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint import MemorySaver
    except ImportError:
        # Fallback for older versions
        class MemorySaver:
            def __init__(self):
                self.storage = {}
            
            def put(self, config, checkpoint, metadata=None):
                thread_id = config.get("configurable", {}).get("thread_id", "default")
                self.storage[thread_id] = {"checkpoint": checkpoint, "metadata": metadata}
            
            def get(self, config):
                thread_id = config.get("configurable", {}).get("thread_id", "default")
                return self.storage.get(thread_id, {"checkpoint": None, "metadata": None})
import json
import re
from datetime import datetime
import warnings
from advanced_analysis import AdvancedSharkTankAnalyzer
from groq_integration import GroqLLM
from config import ANALYSIS_CONFIG, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL, LANGFUSE_ENABLED
warnings.filterwarnings('ignore')

class AnalysisState(TypedDict):
    """State for the analysis workflow"""
    datasets: Dict[str, pd.DataFrame]
    user_query: str
    query_type: str
    country_insights: List[Dict]
    shark_profiles: Dict
    industry_analysis: Dict
    pitch_analysis: Dict
    success_prediction: Dict
    recommendations: List[str]
    final_report: str
    visualizations: Dict
    error_messages: List[str]

class SharkTankAnalyzer:
    """Main analyzer class using LangGraph with Langfuse observability"""
    
    def __init__(self):
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.advanced_analyzer = AdvancedSharkTankAnalyzer()
        self.groq_llm = GroqLLM()
        
        # Initialize Langfuse if enabled
        self.langfuse = None
        if LANGFUSE_ENABLED and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
            try:
                self.langfuse = Langfuse(
                    secret_key=LANGFUSE_SECRET_KEY,
                    public_key=LANGFUSE_PUBLIC_KEY,
                    host=LANGFUSE_BASE_URL
                )
                print("✅ Langfuse initialized in SharkTankAnalyzer")
            except Exception as e:
                print(f"⚠️ Langfuse initialization failed in SharkTankAnalyzer: {e}")
                self.langfuse = None
    
    def _trace_workflow_node(self, node_name: str, func):
        """Decorator to trace workflow node execution"""
        def wrapper(state: AnalysisState) -> AnalysisState:
            if not self.langfuse:
                return func(state)
            
            start_time = time.time()
            trace = None
            
            try:
                # Create trace for this node
                trace = self.langfuse.trace(
                    name=f"workflow_node_{node_name}",
                    input={
                        "node_name": node_name,
                        "state_keys": list(state.keys()) if state else [],
                        "query_type": state.get("query_type", "unknown"),
                        "user_query": state.get("user_query", "")[:100] + "..." if len(state.get("user_query", "")) > 100 else state.get("user_query", "")
                    },
                    metadata={
                        "workflow_node": node_name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Execute the node function
                result = func(state)
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Update trace with success
                trace.update(
                    output={
                        "node_completed": True,
                        "result_keys": list(result.keys()) if result else [],
                        "execution_time": execution_time
                    },
                    metadata={
                        "execution_time_seconds": execution_time,
                        "success": True,
                        "result_state_size": len(str(result)) if result else 0
                    }
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Update trace with error
                if trace:
                    trace.update(
                        error=str(e),
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": False,
                            "error_type": type(e).__name__
                        }
                    )
                
                # Re-raise the exception
                raise e
        
        return wrapper
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with Langfuse tracing"""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes with tracing
        workflow.add_node("query_classifier", self._trace_workflow_node("query_classifier", self._classify_query))
        workflow.add_node("data_loader", self._trace_workflow_node("data_loader", self._load_data))
        workflow.add_node("country_analyzer", self._trace_workflow_node("country_analyzer", self._analyze_countries))
        workflow.add_node("shark_profiler", self._trace_workflow_node("shark_profiler", self._profile_sharks))
        workflow.add_node("industry_analyzer", self._trace_workflow_node("industry_analyzer", self._analyze_industries))
        workflow.add_node("pitch_evaluator", self._trace_workflow_node("pitch_evaluator", self._evaluate_pitch))
        workflow.add_node("groq_analyzer", self._trace_workflow_node("groq_analyzer", self._groq_analysis))
        workflow.add_node("ml_analyzer", self._trace_workflow_node("ml_analyzer", self._ml_analysis))
        workflow.add_node("success_predictor", self._trace_workflow_node("success_predictor", self._predict_success))
        workflow.add_node("recommendation_engine", self._trace_workflow_node("recommendation_engine", self._generate_recommendations))
        workflow.add_node("report_generator", self._trace_workflow_node("report_generator", self._generate_report))
        workflow.add_node("visualization_creator", self._trace_workflow_node("visualization_creator", self._create_visualizations))
        
        # Set entry point
        workflow.set_entry_point("query_classifier")
        
        # Add edges
        workflow.add_edge("query_classifier", "data_loader")
        workflow.add_edge("data_loader", "country_analyzer")
        workflow.add_edge("country_analyzer", "shark_profiler")
        workflow.add_edge("shark_profiler", "industry_analyzer")
        workflow.add_edge("industry_analyzer", "pitch_evaluator")
        workflow.add_edge("pitch_evaluator", "groq_analyzer")
        workflow.add_edge("groq_analyzer", "ml_analyzer")
        workflow.add_edge("ml_analyzer", "success_predictor")
        workflow.add_edge("success_predictor", "recommendation_engine")
        workflow.add_edge("recommendation_engine", "visualization_creator")
        workflow.add_edge("visualization_creator", "report_generator")
        workflow.add_edge("report_generator", END)
        
        return workflow
    
    def _classify_query(self, state: AnalysisState) -> AnalysisState:
        """Classify the type of query with improved detection"""
        query = state.get("user_query", "").lower()
        
        # Shark analysis keywords
        shark_keywords = ["shark", "investor", "barbara", "mark", "lori", "robert", "daymond", "kevin", 
                         "namita", "vineeta", "anupam", "aman", "peyush", "riteshi", "amit",
                         "steve", "janine", "andrew", "naomi", "glen"]
        
        # Country comparison keywords
        country_keywords = ["country", "us", "india", "australia", "compare", "comparison", "versus", "vs"]
        
        # Industry analysis keywords
        industry_keywords = ["industry", "sector", "food", "tech", "health", "fashion", "beauty", "education", 
                           "automotive", "sports", "lifestyle", "medical", "business category"]
        
        # Pitch evaluation keywords
        pitch_keywords = ["pitch", "evaluate", "analyze", "idea", "startup", "business", "company", 
                         "asking", "equity", "investment", "funding", "valuation"]
        
        # Success prediction keywords
        success_keywords = ["success", "predict", "probability", "chance", "likely", "odds", "potential"]
        
        # Data analysis keywords
        data_keywords = ["data", "statistics", "trends", "patterns", "analysis", "insights", "most successful", 
                        "best performing", "top", "average", "rate"]
        
        if any(word in query for word in shark_keywords):
            query_type = "shark_analysis"
        elif any(word in query for word in country_keywords):
            query_type = "country_comparison"
        elif any(word in query for word in industry_keywords):
            query_type = "industry_analysis"
        elif any(word in query for word in pitch_keywords):
            query_type = "pitch_evaluation"
        elif any(word in query for word in success_keywords):
            query_type = "success_prediction"
        elif any(word in query for word in data_keywords):
            query_type = "data_analysis"
        else:
            query_type = "general_analysis"
        
        state["query_type"] = query_type
        return state
    
    def _load_data(self, state: AnalysisState) -> AnalysisState:
        """Load and prepare datasets"""
        try:
            # Load datasets
            us_data = pd.read_csv("Shark Tank US dataset.csv")
            india_data = pd.read_csv("Shark Tank India.csv")
            aus_data = pd.read_csv("Shark Tank Australia dataset.csv")
            merged_data = pd.read_csv("shark_tank_merged.csv")
            
            # Clean and standardize data
            datasets = {
                "us": self._clean_dataset(us_data, "US"),
                "india": self._clean_dataset(india_data, "India"),
                "australia": self._clean_dataset(aus_data, "Australia"),
                "merged": merged_data
            }
            
            # Convert DataFrames to dictionaries for serialization
            serializable_datasets = {}
            for key, df in datasets.items():
                serializable_datasets[key] = {
                    "data": df.to_dict('records'),
                    "columns": df.columns.tolist(),
                    "shape": df.shape
                }
            
            state["datasets"] = serializable_datasets
            state["error_messages"] = []
            
        except Exception as e:
            state["error_messages"] = [f"Error loading data: {str(e)}"]
            
        return state
    
    def _deserialize_datasets(self, serializable_datasets: Dict) -> Dict[str, pd.DataFrame]:
        """Convert serialized datasets back to DataFrames"""
        datasets = {}
        for key, data_info in serializable_datasets.items():
            df = pd.DataFrame(data_info["data"])
            datasets[key] = df
        return datasets
    
    def _clean_dataset(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Clean and standardize dataset"""
        df = df.copy()
        df['country'] = country
        
        # Standardize column names
        if 'Original Ask Amount' in df.columns:
            df['ask_amount'] = pd.to_numeric(df['Original Ask Amount'], errors='coerce')
        elif 'Ask Amount' in df.columns:
            df['ask_amount'] = pd.to_numeric(df['Ask Amount'], errors='coerce')
        
        if 'Original Offered Equity' in df.columns:
            df['equity_offered'] = pd.to_numeric(df['Original Offered Equity'], errors='coerce')
        elif 'Offered Equity' in df.columns:
            df['equity_offered'] = pd.to_numeric(df['Offered Equity'], errors='coerce')
        
        # Create got_deal column based on deal amount
        if 'Got Deal' in df.columns:
            df['got_deal'] = df['Got Deal'].astype(int)
        elif 'Total Deal Amount' in df.columns:
            df['deal_amount'] = pd.to_numeric(df['Total Deal Amount'], errors='coerce')
            df['got_deal'] = (df['deal_amount'] > 0).astype(int)
        else:
            df['got_deal'] = 0  # Default to no deal if no deal information
        
        if 'Total Deal Amount' in df.columns:
            df['deal_amount'] = pd.to_numeric(df['Total Deal Amount'], errors='coerce')
        
        if 'Industry' in df.columns:
            df['industry'] = df['Industry'].str.title()
        elif 'Business Category' in df.columns:
            df['industry'] = df['Business Category'].str.title()
        
        if 'Pitchers Gender' in df.columns:
            df['gender'] = df['Pitchers Gender']
        elif 'Male Presenters' in df.columns:
            df['gender'] = df.apply(lambda x: 'Male' if x['Male Presenters'] > 0 else 'Female', axis=1)
        elif 'Gender' in df.columns:
            df['gender'] = df['Gender']
        else:
            df['gender'] = 'Unknown'
        
        # Fill missing values
        df['ask_amount'] = df['ask_amount'].fillna(0)
        df['equity_offered'] = df['equity_offered'].fillna(0)
        df['industry'] = df['industry'].fillna('Unknown')
        
        return df
    
    def _analyze_countries(self, state: AnalysisState) -> AnalysisState:
        """Analyze patterns across countries"""
        datasets = self._deserialize_datasets(state["datasets"])
        insights = []
        
        for country, df in datasets.items():
            if country == "merged":
                continue
                
            insight = {
                "country": country.title(),
                "total_pitches": int(len(df)),
                "success_rate": float(df['got_deal'].mean()) if 'got_deal' in df.columns else 0.0,
                "avg_ask_amount": float(df['ask_amount'].mean()) if 'ask_amount' in df.columns else 0.0,
                "avg_equity_offered": float(df['equity_offered'].mean()) if 'equity_offered' in df.columns else 0.0,
                "top_industries": {k: int(v) for k, v in df['industry'].value_counts().head(3).to_dict().items()} if 'industry' in df.columns else {},
                "gender_distribution": {k: int(v) for k, v in df['gender'].value_counts().to_dict().items()} if 'gender' in df.columns else {}
            }
            insights.append(insight)
        
        state["country_insights"] = insights
        return state
    
    def _profile_sharks(self, state: AnalysisState) -> AnalysisState:
        """Profile individual shark investment strategies"""
        datasets = self._deserialize_datasets(state["datasets"])
        shark_profiles = {}
        
        # US Sharks
        us_data = datasets.get("us", pd.DataFrame())
        if not us_data.empty:
            us_sharks = {
                "Barbara Corcoran": self._analyze_shark_investments(us_data, "Barbara Corcoran"),
                "Mark Cuban": self._analyze_shark_investments(us_data, "Mark Cuban"),
                "Lori Greiner": self._analyze_shark_investments(us_data, "Lori Greiner"),
                "Robert Herjavec": self._analyze_shark_investments(us_data, "Robert Herjavec"),
                "Daymond John": self._analyze_shark_investments(us_data, "Daymond John"),
                "Kevin O Leary": self._analyze_shark_investments(us_data, "Kevin O Leary")
            }
            shark_profiles["US"] = us_sharks
        
        # India Sharks
        india_data = datasets.get("india", pd.DataFrame())
        if not india_data.empty:
            india_sharks = {
                "Namita": self._analyze_shark_investments(india_data, "Namita"),
                "Vineeta": self._analyze_shark_investments(india_data, "Vineeta"),
                "Anupam": self._analyze_shark_investments(india_data, "Anupam"),
                "Aman": self._analyze_shark_investments(india_data, "Aman"),
                "Peyush": self._analyze_shark_investments(india_data, "Peyush"),
                "Ritesh": self._analyze_shark_investments(india_data, "Ritesh"),
                "Amit": self._analyze_shark_investments(india_data, "Amit")
            }
            shark_profiles["India"] = india_sharks
        
        # Australia Sharks
        aus_data = datasets.get("australia", pd.DataFrame())
        if not aus_data.empty:
            aus_sharks = {
                "Steve": self._analyze_shark_investments(aus_data, "Steve"),
                "Janine": self._analyze_shark_investments(aus_data, "Janine"),
                "Andrew": self._analyze_shark_investments(aus_data, "Andrew"),
                "Naomi": self._analyze_shark_investments(aus_data, "Naomi"),
                "Glen": self._analyze_shark_investments(aus_data, "Glen")
            }
            shark_profiles["Australia"] = aus_sharks
        
        state["shark_profiles"] = shark_profiles
        return state
    
    def _analyze_shark_investments(self, df: pd.DataFrame, shark_name: str) -> Dict:
        """Analyze individual shark's investment patterns"""
        shark_cols = [col for col in df.columns if shark_name in col and "Investment Amount" in col]
        
        if not shark_cols:
            return {"total_investments": 0, "avg_investment": 0, "preferred_industries": {}}
        
        shark_col = shark_cols[0]
        investments = df[df[shark_col] > 0]
        
        return {
            "total_investments": int(len(investments)),
            "total_amount": float(investments[shark_col].sum()) if not investments.empty else 0.0,
            "avg_investment": float(investments[shark_col].mean()) if not investments.empty else 0.0,
            "preferred_industries": {k: int(v) for k, v in investments['industry'].value_counts().head(3).to_dict().items()} if 'industry' in investments.columns and not investments.empty else {}
        }
    
    def _analyze_industries(self, state: AnalysisState) -> AnalysisState:
        """Analyze industry trends and success patterns"""
        datasets = self._deserialize_datasets(state["datasets"])
        industry_analysis = {}
        
        for country, df in datasets.items():
            if country == "merged" or df.empty:
                continue
                
            if 'industry' in df.columns and 'got_deal' in df.columns:
                industry_stats = df.groupby('industry').agg({
                    'got_deal': ['count', 'sum', 'mean'],
                    'ask_amount': 'mean',
                    'equity_offered': 'mean'
                }).round(2)
                
                industry_stats.columns = ['total_pitches', 'successful_deals', 'success_rate', 'avg_ask', 'avg_equity']
                industry_stats = industry_stats.sort_values('success_rate', ascending=False)
                
                # Convert to serializable format
                serializable_stats = {}
                for industry, row in industry_stats.head(10).iterrows():
                    serializable_stats[industry] = {
                        'total_pitches': int(row['total_pitches']),
                        'successful_deals': int(row['successful_deals']),
                        'success_rate': float(row['success_rate']),
                        'avg_ask': float(row['avg_ask']),
                        'avg_equity': float(row['avg_equity'])
                    }
                
                industry_analysis[country] = serializable_stats
        
        state["industry_analysis"] = industry_analysis
        return state
    
    def _evaluate_pitch(self, state: AnalysisState) -> AnalysisState:
        """Evaluate a specific pitch or idea"""
        query = state.get("user_query", "")
        pitch_analysis = {
            "query_analysis": self._extract_pitch_elements(query),
            "success_factors": self._identify_success_factors(state),
            "risk_factors": self._identify_risk_factors(query),
            "recommendations": []
        }
        
        state["pitch_analysis"] = pitch_analysis
        return state
    
    def _groq_analysis(self, state: AnalysisState) -> AnalysisState:
        """Perform advanced analysis using Groq LLM"""
        try:
            pitch_elements = state["pitch_analysis"]["query_analysis"]
            user_query = state.get("user_query", "")
            
            # Prepare context for LLM analysis
            context = {
                "pitch_elements": pitch_elements,
                "country_insights": state.get("country_insights", []),
                "industry_analysis": state.get("industry_analysis", {}),
                "shark_profiles": state.get("shark_profiles", {})
            }
            
            # Get LLM analysis
            llm_analysis = self.groq_llm.analyze_pitch(user_query, context)
            state["groq_analysis"] = llm_analysis
            
            # Generate recommendations using LLM
            if state.get("success_prediction"):
                llm_recommendations = self.groq_llm.generate_recommendations({
                    "pitch_analysis": state["pitch_analysis"],
                    "success_prediction": state["success_prediction"],
                    "context": context
                })
                state["llm_recommendations"] = llm_recommendations
            
            # Generate shark-specific recommendations
            if state.get("shark_profiles"):
                shark_recommendations = self.groq_llm.generate_shark_recommendations(
                    pitch_elements, 
                    state["shark_profiles"]
                )
                state["shark_recommendations"] = shark_recommendations
            
        except Exception as e:
            state["error_messages"] = state.get("error_messages", []) + [f"Groq Analysis error: {str(e)}"]
        
        return state
    
    def _ml_analysis(self, state: AnalysisState) -> AnalysisState:
        """Perform advanced ML analysis"""
        try:
            datasets = self._deserialize_datasets(state["datasets"])
            
            # Train ML models if not already trained
            if not hasattr(self.advanced_analyzer, 'models') or not self.advanced_analyzer.models:
                ml_results = self.advanced_analyzer.train_success_predictor(datasets)
                state["ml_analysis"] = ml_results
            
            # Analyze investment patterns
            patterns = self.advanced_analyzer.analyze_investment_patterns(datasets)
            state["investment_patterns"] = patterns
            
            # Generate insights
            insights = self.advanced_analyzer.generate_insights(datasets)
            state["ml_insights"] = insights
            
        except Exception as e:
            state["error_messages"] = state.get("error_messages", []) + [f"ML Analysis error: {str(e)}"]
        
        return state
    
    def _extract_pitch_elements(self, query: str) -> Dict:
        """Extract key elements from pitch query"""
        elements = {
            "industry": None,
            "ask_amount": None,
            "equity_offered": None,
            "description": query
        }
        
        # Extract industry
        industries = ["food", "tech", "health", "fashion", "education", "automotive", "beauty", "sports"]
        for industry in industries:
            if industry in query.lower():
                elements["industry"] = industry.title()
                break
        
        # Extract amounts
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:k|thousand|million|billion)?'
        amounts = re.findall(amount_pattern, query, re.IGNORECASE)
        if amounts:
            elements["ask_amount"] = float(amounts[0].replace(',', ''))
        
        # Extract equity
        equity_pattern = r'(\d+(?:\.\d+)?)\s*%'
        equity = re.findall(equity_pattern, query)
        if equity:
            elements["equity_offered"] = float(equity[0])
        
        return elements
    
    def _identify_success_factors(self, state: AnalysisState) -> List[str]:
        """Identify key success factors from data"""
        factors = []
        datasets = self._deserialize_datasets(state["datasets"])
        
        # Analyze successful deals
        for country, df in datasets.items():
            if country == "merged" or df.empty or 'got_deal' not in df.columns:
                continue
                
            successful = df[df['got_deal'] == 1]
            if len(successful) > 0:
                # Industry success rates
                if 'industry' in successful.columns:
                    top_industries = successful['industry'].value_counts().head(3)
                    factors.append(f"Top successful industries in {country}: {', '.join(top_industries.index)}")
                
                # Equity range
                if 'equity_offered' in successful.columns:
                    avg_equity = successful['equity_offered'].mean()
                    factors.append(f"Average equity offered in successful deals: {avg_equity:.1f}%")
        
        return factors
    
    def _identify_risk_factors(self, query: str) -> List[str]:
        """Identify potential risk factors in pitch"""
        risks = []
        
        # High equity ask
        equity_pattern = r'(\d+(?:\.\d+)?)\s*%'
        equity = re.findall(equity_pattern, query)
        if equity and float(equity[0]) > 50:
            risks.append("High equity ask (>50%) may deter investors")
        
        # High valuation
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:k|thousand|million|billion)?'
        amounts = re.findall(amount_pattern, query, re.IGNORECASE)
        if amounts:
            amount = float(amounts[0].replace(',', ''))
            if 'million' in query.lower() and amount > 10:
                risks.append("Very high valuation may be difficult to justify")
        
        return risks
    
    def _predict_success(self, state: AnalysisState) -> AnalysisState:
        """Predict success probability for a pitch with enhanced algorithm"""
        pitch_elements = state["pitch_analysis"]["query_analysis"]
        datasets = self._deserialize_datasets(state["datasets"])
        
        # Try ML prediction first
        ml_prediction = None
        if hasattr(self.advanced_analyzer, 'models') and self.advanced_analyzer.models:
            try:
                ml_prediction = self.advanced_analyzer.predict_pitch_success(pitch_elements)
            except Exception as e:
                print(f"ML prediction failed: {e}")
        
        # Enhanced rule-based prediction
        if ml_prediction is None:
            success_probability = 0.3  # Conservative base probability
            confidence = 0.3
            factors = []
            
            # Industry analysis
            if pitch_elements["industry"]:
                industry_success_rates = []
                for country, df in datasets.items():
                    if country == "merged" or df.empty or 'industry' not in df.columns or 'got_deal' not in df.columns:
                        continue
                    
                    # Find matching industries (case-insensitive, partial match)
                    industry_matches = df[df['industry'].str.contains(pitch_elements["industry"], case=False, na=False)]
                    if len(industry_matches) > 0:
                        industry_success_rate = industry_matches['got_deal'].mean()
                        industry_success_rates.append(industry_success_rate)
                        factors.append(f"Industry success rate in {country}: {industry_success_rate:.1%}")
                
                if industry_success_rates:
                    avg_industry_rate = np.mean(industry_success_rates)
                    success_probability = (success_probability + avg_industry_rate) / 2
                    confidence += 0.2
            
            # Equity analysis
            if pitch_elements.get("equity_offered") and pitch_elements["equity_offered"] is not None:
                equity = pitch_elements["equity_offered"]
                if 10 <= equity <= 30:
                    success_probability += 0.15
                    factors.append("Equity ask in optimal range (10-30%)")
                elif 5 <= equity < 10:
                    success_probability += 0.05
                    factors.append("Low equity ask may indicate undervaluation")
                elif 30 < equity <= 50:
                    success_probability -= 0.05
                    factors.append("High equity ask may deter investors")
                elif equity > 50:
                    success_probability -= 0.2
                    factors.append("Very high equity ask likely to be rejected")
                confidence += 0.1
            
            # Ask amount analysis
            if pitch_elements.get("ask_amount") and pitch_elements["ask_amount"] is not None:
                ask_amount = pitch_elements["ask_amount"]
                # Get average ask amounts from successful deals
                successful_asks = []
                for country, df in datasets.items():
                    if country == "merged" or df.empty or 'ask_amount' not in df.columns or 'got_deal' not in df.columns:
                        continue
                    successful_deals = df[df['got_deal'] == 1]
                    if len(successful_deals) > 0:
                        avg_successful_ask = successful_deals['ask_amount'].mean()
                        if not pd.isna(avg_successful_ask):
                            successful_asks.append(avg_successful_ask)
                
                if successful_asks:
                    avg_successful_ask = np.mean(successful_asks)
                    if ask_amount <= avg_successful_ask * 1.5:
                        success_probability += 0.1
                        factors.append(f"Ask amount reasonable compared to successful deals")
                    elif ask_amount > avg_successful_ask * 3:
                        success_probability -= 0.15
                        factors.append(f"Ask amount very high compared to successful deals")
                    confidence += 0.1
            
            # Gender analysis
            for country, df in datasets.items():
                if country == "merged" or df.empty or 'gender' not in df.columns or 'got_deal' not in df.columns:
                    continue
                
                gender_success = df.groupby('gender')['got_deal'].mean()
                if len(gender_success) > 1:
                    factors.append(f"Gender success rates in {country}: {gender_success.to_dict()}")
                    confidence += 0.05
            
            # Country-specific analysis
            country_insights = state.get("country_insights", [])
            for insight in country_insights:
                if insight.get("success_rate", 0) > 0.6:
                    success_probability += 0.05
                    factors.append(f"High success rate in {insight['country']}")
                confidence += 0.05
            
            success_probability = max(0.1, min(0.95, success_probability))
            confidence = min(0.9, confidence)
            
            # Update factors in state
            state["pitch_analysis"]["success_factors"] = factors
            
        else:
            success_probability = ml_prediction["probability"]
            confidence = ml_prediction["confidence"]
            factors = state["pitch_analysis"]["success_factors"]
        
        # Determine success level with more nuanced thresholds
        if success_probability >= 0.7:
            success_level = "High"
        elif success_probability >= 0.5:
            success_level = "Medium"
        elif success_probability >= 0.3:
            success_level = "Low"
        else:
            success_level = "Very Low"
        
        # Add risk assessment
        risks = []
        if pitch_elements.get("equity_offered") and pitch_elements["equity_offered"] > 50:
            risks.append("Very high equity ask (>50%)")
        if pitch_elements.get("ask_amount") and pitch_elements["ask_amount"] > 1000000:  # $1M+
            risks.append("High valuation may be difficult to justify")
        if not pitch_elements.get("industry"):
            risks.append("Unclear industry focus")
        
        state["success_prediction"] = {
            "probability": success_probability,
            "confidence": confidence,
            "success_level": success_level,
            "ml_prediction": ml_prediction is not None,
            "factors": factors,
            "risks": risks
        }
        
        return state
    
    def _generate_recommendations(self, state: AnalysisState) -> AnalysisState:
        """Generate actionable recommendations"""
        recommendations = []
        pitch_analysis = state["pitch_analysis"]
        success_prediction = state["success_prediction"]
        
        # Add LLM recommendations if available
        if "llm_recommendations" in state:
            recommendations.extend(state["llm_recommendations"])
        
        # Industry recommendations
        if pitch_analysis["query_analysis"]["industry"]:
            recommendations.append(f"Focus on {pitch_analysis['query_analysis']['industry']} industry - shows strong success rates")
        
        # Equity recommendations
        if pitch_analysis["query_analysis"]["equity_offered"]:
            equity = pitch_analysis["query_analysis"]["equity_offered"]
            if equity > 30:
                recommendations.append("Consider reducing equity ask to 10-30% range for better investor appeal")
            elif equity < 10:
                recommendations.append("Consider increasing equity offer to show commitment")
        
        # Shark recommendations
        shark_profiles = state["shark_profiles"]
        if shark_profiles:
            recommendations.append("Research shark preferences before pitching - each has different investment focus")
        
        # General recommendations
        if success_prediction["probability"] < 0.4:
            recommendations.append("Consider refining your pitch and business model before approaching investors")
        elif success_prediction["probability"] > 0.7:
            recommendations.append("Your pitch shows strong potential - prepare for multiple shark interest")
        
        # Remove duplicates and limit recommendations
        recommendations = list(dict.fromkeys(recommendations))[:10]
        state["recommendations"] = recommendations
        return state
    
    def _create_visualizations(self, state: AnalysisState) -> AnalysisState:
        """Create visualization data for charts"""
        visualizations = {}
        
        # Country comparison
        if state["country_insights"]:
            visualizations["country_comparison"] = {
                "countries": [insight["country"] for insight in state["country_insights"]],
                "success_rates": [insight["success_rate"] for insight in state["country_insights"]],
                "avg_ask_amounts": [insight["avg_ask_amount"] for insight in state["country_insights"]]
            }
        
        # Industry analysis
        if state["industry_analysis"]:
            visualizations["industry_analysis"] = state["industry_analysis"]
        
        # Shark profiles
        if state["shark_profiles"]:
            visualizations["shark_profiles"] = state["shark_profiles"]
        
        state["visualizations"] = visualizations
        return state
    
    def _generate_report(self, state: AnalysisState) -> AnalysisState:
        """Generate comprehensive analysis report"""
        report_sections = []
        
        # Executive Summary
        success_pred = state["success_prediction"]
        report_sections.append(f"""
# Shark Tank Analysis Report

## Executive Summary
**Success Probability: {success_pred['probability']:.1%}**
**Confidence Level: {success_pred['confidence']:.1%}**

Based on historical Shark Tank data analysis, your pitch shows {'strong' if success_pred['probability'] > 0.6 else 'moderate' if success_pred['probability'] > 0.4 else 'limited'} potential for success.
""")
        
        # Country Insights
        if state["country_insights"]:
            report_sections.append("## Market Analysis by Country")
            for insight in state["country_insights"]:
                report_sections.append(f"""
### {insight['country']}
- **Total Pitches:** {insight['total_pitches']}
- **Success Rate:** {insight['success_rate']:.1%}
- **Average Ask Amount:** ${insight['avg_ask_amount']:,.0f}
- **Average Equity Offered:** {insight['avg_equity_offered']:.1f}%
""")
        
        # Industry Analysis
        if state["industry_analysis"]:
            report_sections.append("## Industry Performance")
            for country, industries in state["industry_analysis"].items():
                report_sections.append(f"### {country.title()}")
                for industry, stats in list(industries.items())[:5]:
                    report_sections.append(f"- **{industry}:** {stats['success_rate']:.1%} success rate ({stats['total_pitches']} pitches)")
        
        # LLM Analysis - Generate if not present
        if "groq_analysis" not in state:
            try:
                # Generate Groq analysis on the fly
                pitch_elements = state["pitch_analysis"]["query_analysis"]
                user_query = state.get("user_query", "")
                context = {
                    "pitch_elements": pitch_elements,
                    "country_insights": state.get("country_insights", []),
                    "industry_analysis": state.get("industry_analysis", {}),
                    "shark_profiles": state.get("shark_profiles", {})
                }
                llm_analysis = self.groq_llm.analyze_pitch(user_query, context)
                state["groq_analysis"] = llm_analysis
            except Exception as e:
                state["error_messages"] = state.get("error_messages", []) + [f"Groq Analysis error: {str(e)}"]
        
        if "groq_analysis" in state:
            report_sections.append("## AI Analysis")
            report_sections.append(state["groq_analysis"]["analysis"])
        
        # Shark-Specific Recommendations
        if "shark_recommendations" in state and state["shark_recommendations"]:
            report_sections.append("## Shark-Specific Recommendations")
            for shark, rec in state["shark_recommendations"].items():
                report_sections.append(f"### {shark}")
                report_sections.append(rec)
        
        # Recommendations
        if state["recommendations"]:
            report_sections.append("## Recommendations")
            for i, rec in enumerate(state["recommendations"], 1):
                report_sections.append(f"{i}. {rec}")
        
        # Risk Factors
        if success_pred["risks"]:
            report_sections.append("## Risk Factors")
            for risk in success_pred["risks"]:
                report_sections.append(f"- {risk}")
        
        state["final_report"] = "\n".join(report_sections)
        return state
    
    def analyze(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Run the complete analysis workflow with Langfuse tracing"""
        workflow_trace = None
        start_time = time.time()
        
        try:
            # Create main workflow trace if Langfuse is enabled
            if self.langfuse:
                workflow_trace = self.langfuse.trace(
                    name="shark_tank_analysis_workflow",
                    input={
                        "user_query": query,
                        "thread_id": thread_id,
                        "query_length": len(query)
                    },
                    metadata={
                        "workflow_type": "complete_analysis",
                        "timestamp": datetime.now().isoformat(),
                        "thread_id": thread_id
                    }
                )
            
            # Execute the workflow
            result = self.app.invoke(
                {"user_query": query},
                {"configurable": {"thread_id": thread_id}}
            )
            
            # Calculate total execution time
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            # Update workflow trace with success
            if workflow_trace:
                workflow_trace.update(
                    output={
                        "analysis_completed": True,
                        "result_keys": list(result.keys()) if result else [],
                        "total_execution_time": total_execution_time,
                        "success_prediction": result.get("success_prediction", {}),
                        "query_type": result.get("query_type", "unknown")
                    },
                    metadata={
                        "total_execution_time_seconds": total_execution_time,
                        "success": True,
                        "result_size": len(str(result)) if result else 0,
                        "has_success_prediction": "success_prediction" in result,
                        "has_recommendations": "recommendations" in result,
                        "has_visualizations": "visualizations" in result
                    }
                )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            # Update workflow trace with error
            if workflow_trace:
                workflow_trace.update(
                    error=str(e),
                    metadata={
                        "total_execution_time_seconds": total_execution_time,
                        "success": False,
                        "error_type": type(e).__name__
                    }
                )
            
            return {
                "error": str(e),
                "final_report": f"Error in analysis: {str(e)}"
            }
