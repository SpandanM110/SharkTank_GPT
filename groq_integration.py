"""
Groq LLM Integration for Shark Tank AI Analyzer
"""
import os
from typing import List, Dict, Any, Optional
from groq import Groq
from config import GROQ_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL, LANGFUSE_ENABLED
from langfuse import Langfuse
import time

class GroqLLM:
    """Groq LLM integration class with Langfuse observability"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client and Langfuse"""
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Initialize Langfuse if enabled
        self.langfuse = None
        if LANGFUSE_ENABLED and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
            try:
                self.langfuse = Langfuse(
                    secret_key=LANGFUSE_SECRET_KEY,
                    public_key=LANGFUSE_PUBLIC_KEY,
                    host=LANGFUSE_BASE_URL
                )
                print("✅ Langfuse initialized successfully")
            except Exception as e:
                print(f"⚠️ Langfuse initialization failed: {e}")
                self.langfuse = None
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.7,
                         max_tokens: int = 1024,
                         stream: bool = False,
                         trace_name: str = "groq_llm_call",
                         trace_metadata: Dict[str, Any] = None) -> str:
        """Generate response using Groq LLM with Langfuse tracing"""
        
        # Start Langfuse trace if enabled
        trace = None
        if self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name=trace_name,
                    input={
                        "messages": messages,
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream
                    },
                    metadata=trace_metadata or {}
                )
            except Exception as e:
                print(f"⚠️ Langfuse trace creation failed: {e}")
        
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=stream,
                stop=None
            )
            
            if stream:
                response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
            else:
                response = completion.choices[0].message.content
            
            # Calculate metrics
            end_time = time.time()
            latency = end_time - start_time
            
            # Update Langfuse trace with success
            if trace:
                try:
                    trace.update(
                        output=response,
                        metadata={
                            "latency_seconds": latency,
                            "model": self.model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "success": True,
                            **(trace_metadata or {})
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Langfuse trace update failed: {e}")
            
            return response
                
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            # Update Langfuse trace with error
            if trace:
                try:
                    trace.update(
                        error=str(e),
                        metadata={
                            "latency_seconds": latency,
                            "model": self.model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "success": False,
                            "error_type": type(e).__name__,
                            **(trace_metadata or {})
                        }
                    )
                except Exception as trace_error:
                    print(f"⚠️ Langfuse error trace update failed: {trace_error}")
            
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def analyze_pitch(self, pitch_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze a pitch using the LLM with Langfuse tracing"""
        system_prompt = """You are an expert Shark Tank investment analyst with deep knowledge of startup evaluation, market analysis, and investment patterns. Your role is to provide comprehensive analysis of startup pitches based on historical Shark Tank data and investment trends.

Key areas to analyze:
1. Market potential and industry trends
2. Business model viability
3. Financial projections and valuation
4. Competitive advantages
5. Team strength and execution capability
6. Investment risks and opportunities
7. Success probability based on historical patterns

Provide detailed, actionable insights that would help both entrepreneurs and investors make informed decisions."""

        user_prompt = f"""
Analyze this startup pitch:

Pitch Description: {pitch_description}

Additional Context: {context or "No additional context provided"}

Please provide:
1. Overall assessment (1-10 scale)
2. Key strengths
3. Potential weaknesses/risks
4. Market opportunity analysis
5. Investment recommendation
6. Suggested improvements
7. Success probability estimate
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Create Langfuse prompt if enabled
        if self.langfuse:
            try:
                self.langfuse.create_prompt(
                    name="pitch_analysis_prompt",
                    prompt=system_prompt,
                    labels=["shark_tank", "pitch_analysis", "investment"]
                )
            except Exception as e:
                print(f"⚠️ Langfuse prompt creation failed: {e}")
        
        response = self.generate_response(
            messages, 
            temperature=0.7, 
            max_tokens=1024,
            trace_name="pitch_analysis",
            trace_metadata={
                "analysis_type": "pitch_evaluation",
                "context_keys": list(context.keys()) if context else [],
                "pitch_length": len(pitch_description)
            }
        )
        
        return {
            "analysis": response,
            "model": self.model,
            "timestamp": str(pd.Timestamp.now())
        }
    
    def generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis with Langfuse tracing"""
        system_prompt = """You are a startup advisor specializing in Shark Tank preparation. Based on the analysis data provided, generate specific, actionable recommendations for improving the pitch and business model."""

        user_prompt = f"""
Based on this analysis data, generate 5-7 specific recommendations:

Analysis Data: {analysis_data}

Provide recommendations that are:
1. Specific and actionable
2. Based on historical Shark Tank success patterns
3. Focused on improving investment appeal
4. Practical and implementable
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Create Langfuse prompt if enabled
        if self.langfuse:
            try:
                self.langfuse.create_prompt(
                    name="recommendations_prompt",
                    prompt=system_prompt,
                    labels=["shark_tank", "recommendations", "advice"]
                )
            except Exception as e:
                print(f"⚠️ Langfuse prompt creation failed: {e}")
        
        response = self.generate_response(
            messages, 
            temperature=0.8, 
            max_tokens=512,
            trace_name="generate_recommendations",
            trace_metadata={
                "analysis_type": "recommendations",
                "analysis_data_keys": list(analysis_data.keys()) if analysis_data else []
            }
        )
        
        # Parse recommendations into a list
        recommendations = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '-', '•')) or 
                       any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'consider', 'improve'])):
                # Clean up the line
                clean_line = line.lstrip('123456789.-• ').strip()
                if clean_line:
                    recommendations.append(clean_line)
        
        return recommendations[:7]  # Limit to 7 recommendations
    
    def compare_with_historical_data(self, pitch_data: Dict[str, Any], historical_insights: Dict[str, Any]) -> str:
        """Compare pitch with historical Shark Tank data with Langfuse tracing"""
        system_prompt = """You are a data analyst specializing in Shark Tank investment patterns. Compare the given pitch with historical data to identify patterns, similarities, and differences that could impact investment success."""

        user_prompt = f"""
Compare this pitch with historical Shark Tank data:

Pitch Data: {pitch_data}

Historical Insights: {historical_insights}

Provide:
1. Similar successful pitches from history
2. Key differences from successful patterns
3. Industry trends and market timing
4. Shark preferences and investment patterns
5. Risk factors based on historical failures
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate_response(
            messages, 
            temperature=0.6, 
            max_tokens=800,
            trace_name="historical_comparison",
            trace_metadata={
                "analysis_type": "historical_comparison",
                "pitch_data_keys": list(pitch_data.keys()) if pitch_data else [],
                "historical_insights_keys": list(historical_insights.keys()) if historical_insights else []
            }
        )
    
    def generate_shark_recommendations(self, pitch_data: Dict[str, Any], shark_profiles: Dict[str, Any]) -> Dict[str, str]:
        """Generate specific recommendations for each shark with Langfuse tracing"""
        recommendations = {}
        
        for shark_name, profile in shark_profiles.items():
            system_prompt = f"""You are analyzing this pitch specifically for {shark_name}, a Shark Tank investor. Based on their investment history and preferences, provide tailored advice for this entrepreneur."""

            user_prompt = f"""
Shark: {shark_name}
Shark Profile: {profile}

Pitch Data: {pitch_data}

Provide specific advice for approaching {shark_name}:
1. Why this pitch might appeal to them
2. How to tailor the presentation
3. What to emphasize or de-emphasize
4. Potential concerns they might have
5. Negotiation strategy
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.generate_response(
                messages, 
                temperature=0.7, 
                max_tokens=400,
                trace_name=f"shark_recommendation_{shark_name.lower().replace(' ', '_')}",
                trace_metadata={
                    "analysis_type": "shark_specific_recommendation",
                    "shark_name": shark_name,
                    "pitch_data_keys": list(pitch_data.keys()) if pitch_data else []
                }
            )
            recommendations[shark_name] = response
        
        return recommendations

# Import pandas for timestamp
import pandas as pd
