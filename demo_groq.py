#!/usr/bin/env python3
"""
Demo script for Shark Tank AI Analyzer with Groq integration
"""
import sys
import os

def test_groq_integration():
    """Test Groq LLM integration"""
    print("Testing Groq LLM Integration...")
    print("=" * 40)
    
    try:
        from groq_integration import GroqLLM
        
        # Initialize Groq LLM
        llm = GroqLLM()
        print("Groq LLM initialized successfully")
        
        # Test simple analysis
        test_pitch = "I want to pitch a food tech startup asking for $500k for 15% equity"
        print(f"\nTesting with pitch: {test_pitch}")
        
        # Get analysis
        analysis = llm.analyze_pitch(test_pitch)
        print("Analysis completed successfully")
        print(f"Analysis preview: {analysis['analysis'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_full_workflow():
    """Test the complete LangGraph workflow"""
    print("\nTesting Complete Workflow...")
    print("=" * 40)
    
    try:
        from langgraph_workflow import SharkTankAnalyzer
        
        # Initialize analyzer
        analyzer = SharkTankAnalyzer()
        print("Analyzer initialized successfully")
        
        # Test analysis
        test_query = "Analyze my pitch: AI-powered fitness app, $1M ask, 20% equity"
        print(f"\nTesting with query: {test_query}")
        
        result = analyzer.analyze(test_query)
        print("Analysis completed successfully")
        
        # Check if Groq analysis is included
        if "groq_analysis" in result:
            print("Groq LLM analysis included in results")
        else:
            print("Groq LLM analysis not found in results")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main demo function"""
    print("Shark Tank AI Analyzer - Groq Integration Demo")
    print("=" * 60)
    
    # Test Groq integration
    groq_success = test_groq_integration()
    
    # Test full workflow
    workflow_success = test_full_workflow()
    
    print("\n" + "=" * 60)
    print("Demo Results:")
    print(f"Groq Integration: {'PASSED' if groq_success else 'FAILED'}")
    print(f"Full Workflow: {'PASSED' if workflow_success else 'FAILED'}")
    
    if groq_success and workflow_success:
        print("\nAll tests passed! The system is ready to use.")
        print("\nTo start the application:")
        print("  python run_app.py")
        print("  or")
        print("  streamlit run streamlit_app.py")
    else:
        print("\nSome tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
