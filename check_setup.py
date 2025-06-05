#!/usr/bin/env python3
"""
Check file structure and setup for AI Psychologist
"""

import os
from pathlib import Path

def check_file_structure():
    """Check if all required files exist"""
    print("🔍 Checking file structure...")
    
    current_dir = Path(".")
    required_files = [
        "main.py",
        "index.html", 
        "requirements.txt",
        ".env"
    ]
    
    print(f"📁 Current directory: {current_dir.absolute()}")
    print("\n📋 Required files:")
    
    all_good = True
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - NOT FOUND")
            all_good = False
    
    print("\n📁 All files in directory:")
    for item in sorted(current_dir.iterdir()):
        if item.is_file():
            print(f"   📄 {item.name}")
        elif item.is_dir():
            print(f"   📁 {item.name}/")
    
    return all_good

def check_environment():
    """Check environment variables"""
    print("\n⚙️ Checking environment...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Read .env file
        with open(env_file, 'r') as f:
            content = f.read()
        
        if "GEMINI_API_KEY" in content:
            if "your_gemini_api_key_here" in content:
                print("⚠️ GEMINI_API_KEY found but still has placeholder value")
                print("   Please edit .env and add your actual API key")
            else:
                print("✅ GEMINI_API_KEY appears to be set")
        else:
            print("❌ GEMINI_API_KEY not found in .env file")
    else:
        print("❌ .env file not found")
        return False
    
    return True

def check_html_file():
    """Check HTML file content"""
    print("\n📄 Checking index.html...")
    
    html_file = Path("index.html")
    if html_file.exists():
        try:
            content = html_file.read_text(encoding='utf-8')
            if "AI Psychologist" in content:
                print("✅ index.html looks good")
                if "__RTC_CONFIGURATION__" in content:
                    print("✅ RTC configuration placeholder found")
                else:
                    print("⚠️ RTC configuration placeholder not found")
            else:
                print("⚠️ index.html doesn't contain expected content")
        except Exception as e:
            print(f"❌ Error reading index.html: {e}")
    else:
        print("❌ index.html not found")

def main():
    print("🧠 AI Psychologist Setup Check")
    print("=" * 40)
    
    # Check file structure
    files_ok = check_file_structure()
    
    # Check environment
    env_ok = check_environment()
    
    # Check HTML file
    check_html_file()
    
    print("\n" + "=" * 40)
    if files_ok and env_ok:
        print("🎉 Setup looks good! Try running: python main.py")
    else:
        print("⚠️ Some issues found. Please fix them before running the app.")
        
        if not files_ok:
            print("\n💡 Missing files? Make sure you have:")
            print("   - main.py (the main application)")
            print("   - index.html (the web interface)")
            print("   - requirements.txt (dependencies)")
            print("   - .env (environment configuration)")
        
        if not env_ok:
            print("\n💡 Environment issues? Try:")
            print("   1. Copy .env.example to .env")
            print("   2. Edit .env and add your Gemini API key")
            print("   3. Get your key from: https://ai.google.dev/gemini-api/docs/api-key")

if __name__ == "__main__":
    main()
