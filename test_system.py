#!/usr/bin/env python3
"""
Test script for Shark Tank AI Analyzer
"""
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from langgraph_workflow import SharkTankAnalyzer
        from advanced_analysis import AdvancedSharkTankAnalyzer
        import pandas as pd
        import plotly
        import sklearn
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_data_loading():
    """Test if datasets can be loaded"""
    try:
        import pandas as pd
        
        # Check if files exist
        files = [
            "Shark Tank US dataset.csv",
            "Shark Tank India.csv", 
            "Shark Tank Australia dataset.csv",
            "shark_tank_merged.csv"
        ]
        
        for file in files:
            if not os.path.exists(file):
                print(f"Missing file: {file}")
                return False
        
        # Try to load one dataset
        df = pd.read_csv("Shark Tank US dataset.csv")
        print(f"Data loading successful - US dataset has {len(df)} rows")
        return True
        
    except Exception as e:
        print(f"Data loading error: {e}")
        return False

def test_analyzer():
    """Test the analyzer functionality"""
    try:
        from langgraph_workflow import SharkTankAnalyzer
        
        analyzer = SharkTankAnalyzer()
        print("Analyzer initialized successfully")
        
        # Test a simple query
        result = analyzer.analyze("Test query for food tech startup")
        print("Analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"Analyzer error: {e}")
        return False

def main():
    """Run all tests"""
    print("Shark Tank AI Analyzer - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Analyzer Test", test_analyzer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
            print(f"PASSED: {test_name}")
        else:
            print(f"FAILED: {test_name}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready to use.")
        print("\nTo start the application, run:")
        print("  python run_app.py")
        print("  or")
        print("  streamlit run streamlit_app.py")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
