#!/usr/bin/env python3
"""
Script to fix common deployment issues for Railway
"""

import os
import sys
import json
import subprocess

def check_environment():
    """Check environment variables and configuration"""
    print("🔧 Checking deployment environment...")
    
    required_env_vars = [
        "GEMINI_API_KEY",
        "PORT"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ Missing environment variable: {var}")
        else:
            # Don't print the actual API key for security
            if "KEY" in var or "SECRET" in var:
                print(f"✅ {var}: [REDACTED]")
            else:
                print(f"✅ {var}: {os.getenv(var)}")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your Railway dashboard under Variables.")
        return False
    
    return True

def check_port_configuration():
    """Check if port configuration is correct"""
    print("\n🔧 Checking port configuration...")
    
    port = os.getenv("PORT", "8000")
    print(f"Port from environment: {port}")
    
    # Check if port is numeric
    try:
        port_num = int(port)
        if 1000 <= port_num <= 65535:
            print(f"✅ Port {port_num} is valid")
            return True
        else:
            print(f"❌ Port {port_num} is out of valid range (1000-65535)")
            return False
    except ValueError:
        print(f"❌ Port '{port}' is not a valid number")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("\n🔧 Checking Python version...")
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.9 or higher")
        return False

def generate_railway_config():
    """Generate optimized Railway configuration"""
    print("\n🔧 Generating Railway configuration...")
    
    config = {
        "$schema": "https://railway.app/railway.schema.json",
        "build": {
            "builder": "NIXPACKS",
            "buildCommand": "pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt"
        },
        "deploy": {
            "startCommand": "python main.py",
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10,
            "healthcheckPath": "/health",
            "healthcheckTimeout": 300
        }
    }
    
    try:
        with open("railway.json", "w") as f:
            json.dump(config, f, indent=2)
        print("✅ Generated railway.json")
        return True
    except Exception as e:
        print(f"❌ Error generating railway.json: {e}")
        return False

def test_imports():
    """Test critical imports"""
    print("\n🔧 Testing critical imports...")
    
    critical_imports = [
        "fastapi",
        "uvicorn",
        "google.genai",
        "pydantic",
        "dotenv",
        "numpy"
    ]
    
    failed_imports = []
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    # Test FastRTC separately since it might not be available
    try:
        import fastrtc
        print(f"✅ fastrtc (version: {getattr(fastrtc, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"⚠️  fastrtc: {e} (Voice features will be limited)")
    
    return len(failed_imports) == 0

def main():
    print("🚀 Railway Deployment Fix Script")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", check_environment),
        ("Port Configuration", check_port_configuration),
        ("Python Version", check_python_version),
        ("Critical Imports", test_imports),
        ("Railway Config", generate_railway_config)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ All checks passed! Your deployment should work now.")
        print("\n📝 Next steps:")
        print("1. Commit and push your changes to GitHub")
        print("2. Redeploy on Railway")
        print("3. Check the deployment logs")
        print("4. Test the /health endpoint")
    else:
        print("⚠️  Some issues were found. Please fix them before deploying.")
        print("\n🔧 Common fixes:")
        print("1. Set missing environment variables in Railway dashboard")
        print("2. Update requirements.txt with correct package versions")
        print("3. Check Railway build logs for specific errors")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
