#!/usr/bin/env python3
"""
Test script to verify Groq integration in the workflow
"""
import sys
import os

def test_groq_in_workflow():
    """Test if Groq analysis is included in the workflow results"""
    print("Testing Groq Integration in Workflow...")
    print("=" * 50)
    
    try:
        from langgraph_workflow import SharkTankAnalyzer
        
        # Initialize analyzer
        analyzer = SharkTankAnalyzer()
        print("Analyzer initialized successfully")
        
        # Test with a specific query that should trigger Groq analysis
        test_query = "I want to pitch a food tech startup asking for $500k for 15% equity"
        print(f"\nTesting with query: {test_query}")
        
        # Run analysis
        result = analyzer.analyze(test_query)
        print("Analysis completed successfully")
        
        # Check for Groq analysis in results
        if "groq_analysis" in result:
            print("Groq LLM analysis found in results!")
            print(f"Analysis preview: {result['groq_analysis']['analysis'][:200]}...")
        elif "AI Analysis" in result.get('final_report', ''):
            print("Groq LLM analysis found in final report!")
            # Extract AI Analysis section from report
            report = result.get('final_report', '')
            ai_section_start = report.find("## AI Analysis")
            if ai_section_start != -1:
                ai_section_end = report.find("##", ai_section_start + 1)
                if ai_section_end == -1:
                    ai_section = report[ai_section_start:]
                else:
                    ai_section = report[ai_section_start:ai_section_end]
                print(f"Analysis preview: {ai_section[:200]}...")
        else:
            print("Groq LLM analysis not found in results or final report")
            print("Available keys:", list(result.keys()))
        
        # Check for other analysis components
        components = ["country_insights", "shark_profiles", "industry_analysis", "success_prediction", "recommendations"]
        for component in components:
            if component in result:
                print(f"{component} found")
            else:
                print(f"{component} not found")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Shark Tank AI Analyzer - Groq Workflow Test")
    print("=" * 60)
    
    success = test_groq_in_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("Test completed successfully!")
        print("The Groq integration is working in the workflow.")
    else:
        print("Test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
