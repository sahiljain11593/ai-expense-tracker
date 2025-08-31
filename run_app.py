#!/usr/bin/env python3
"""
Simple launcher script for the Expense Tracker app.
Run this script to start the application.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit expense tracker app."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the app directory
    os.chdir(script_dir)
    
    print("🚀 Starting AI-Powered Expense Tracker...")
    print(f"📁 App directory: {script_dir}")
    print("🌐 Opening in browser...")
    print("💡 Tip: Keep this terminal open while using the app")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "transaction_web_app.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Thanks for using the Expense Tracker!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
