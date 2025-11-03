"""
Test script for Langfuse integration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langfuse_config():
    """Test Langfuse configuration"""
    print("🔧 Testing Langfuse Configuration...")
    
    # Check environment variables
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    
    print(f"✅ LANGFUSE_ENABLED: {enabled}")
    print(f"✅ LANGFUSE_SECRET_KEY: {'Set' if secret_key else 'Not Set'}")
    print(f"✅ LANGFUSE_PUBLIC_KEY: {'Set' if public_key else 'Not Set'}")
    print(f"✅ LANGFUSE_BASE_URL: {base_url}")
    
    if enabled and secret_key and public_key:
        print("🎉 Langfuse configuration looks good!")
        return True
    else:
        print("⚠️ Langfuse configuration incomplete")
        return False

def test_groq_integration():
    """Test Groq integration with Langfuse"""
    print("\n🔧 Testing Groq Integration...")
    
    try:
        from groq_integration import GroqLLM
        print("✅ GroqLLM import successful")
        
        # Test initialization
        llm = GroqLLM()
        print("✅ GroqLLM initialization successful")
        
        if llm.langfuse:
            print("✅ Langfuse integration active in GroqLLM")
        else:
            print("⚠️ Langfuse integration not active in GroqLLM")
            
        return True
        
    except Exception as e:
        print(f"❌ Groq integration test failed: {e}")
        return False

def test_langgraph_integration():
    """Test LangGraph integration with Langfuse"""
    print("\n🔧 Testing LangGraph Integration...")
    
    try:
        from langgraph_workflow import SharkTankAnalyzer
        print("✅ SharkTankAnalyzer import successful")
        
        # Test initialization
        analyzer = SharkTankAnalyzer()
        print("✅ SharkTankAnalyzer initialization successful")
        
        if analyzer.langfuse:
            print("✅ Langfuse integration active in SharkTankAnalyzer")
        else:
            print("⚠️ Langfuse integration not active in SharkTankAnalyzer")
            
        return True
        
    except Exception as e:
        print(f"❌ LangGraph integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Langfuse Integration for Shark Tank AI System\n")
    
    config_ok = test_langfuse_config()
    groq_ok = test_groq_integration()
    langgraph_ok = test_langgraph_integration()
    
    print(f"\n📊 Test Results:")
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"Groq Integration: {'✅' if groq_ok else '❌'}")
    print(f"LangGraph Integration: {'✅' if langgraph_ok else '❌'}")
    
    if config_ok and groq_ok and langgraph_ok:
        print("\n🎉 All tests passed! Langfuse integration is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the configuration.")

if __name__ == "__main__":
    main()
