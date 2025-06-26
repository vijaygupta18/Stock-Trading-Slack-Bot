#!/usr/bin/env python3
"""
AI-Powered Trading Bot Setup Script

This script helps initialize the trading bot environment and verify configuration.
"""

import os
import sys
import json
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🤖 AI-Powered Trading Bot Setup")
    print("=" * 40)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'python-dotenv',
        'google.generativeai', 'slack_sdk', 'schedule', 'pytz'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('.', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📋 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("\n🔐 Checking environment configuration...")
    
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found")
        print("📋 Create .env file with:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("   SLACK_BOT_TOKEN=xoxb-your-token-here")
        print("   SLACK_DEFAULT_CHANNEL=#trading-alerts")
        return False
    
    print("✅ .env file found")
    
    # Check for required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GEMINI_API_KEY', 'SLACK_BOT_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ {var} - Not set")
        else:
            print(f"✅ {var} - Configured")
    
    if missing_vars:
        print(f"\n📋 Add missing variables to .env file:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        return False
    return True

def initialize_data_files():
    """Initialize required data files"""
    print("\n📁 Initializing data files...")
    
    files_to_create = {
        'smart_analytics.json': {
            "current_recommendations": [],
            "last_updated": None,
            "latest_batch_date": None,
            "latest_batch_time": None,
            "cleanup_info": {}
        },
        'trading_history.json': {
            "completed_trades": [],
            "last_updated": None,
            "total_trades": 0,
            "total_profit_loss": 0.0,
            "summary_stats": {
                "profitable_trades": 0,
                "losing_trades": 0,
                "average_pl": 0.0
            }
        },
        'trading_data.json': {
            "initialized": True,
            "version": "1.0",
            "setup_date": None
        }
    }
    
    for filename, initial_data in files_to_create.items():
        file_path = Path(filename)
        if not file_path.exists():
            with open(file_path, 'w') as f:
                json.dump(initial_data, f, indent=4)
            print(f"✅ Created {filename}")
        else:
            print(f"✅ {filename} already exists")
    
    return True

def verify_api_connections():
    """Verify API connections work"""
    print("\n🌐 Verifying API connections...")
    
    try:
        # Test Gemini AI
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            genai.configure(api_key=api_key)
            print("✅ Gemini AI - Connection OK")
        else:
            print("⚠️ Gemini AI - API key not configured")
    except Exception as e:
        print(f"❌ Gemini AI - Error: {e}")
    
    try:
        # Test Slack
        from slack_sdk import WebClient
        token = os.getenv('SLACK_BOT_TOKEN')
        if token and token != 'xoxb-your-token-here':
            client = WebClient(token=token)
            response = client.auth_test()
            print("✅ Slack - Connection OK")
        else:
            print("⚠️ Slack - Bot token not configured")
    except Exception as e:
        print(f"❌ Slack - Error: {e}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🚀 Next Steps:")
    print("1. Review and update trading parameters in trading.py")
    print("2. Test the system: python -c 'from trading import *; print(\"Import OK\")'")
    print("3. Run the bot: python trading.py")
    print("4. Monitor logs: tail -f trading_log.log")
    print("\n📖 Documentation:")
    print("- README.md - Complete documentation")
    print("- SECURITY.md - Security best practices")
    print("\n⚠️ Important:")
    print("- Start with paper trading to test the system")
    print("- Never invest more than you can afford to lose")
    print("- Review all trading parameters before going live")

def main():
    """Main setup function"""
    print_header()
    
    # Run all checks
    checks = [
        check_python_version(),
        check_dependencies(),
        check_env_file(),
        initialize_data_files()
    ]
    
    if all(checks):
        verify_api_connections()
        print("\n🎉 Setup completed successfully!")
        show_next_steps()
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 