#!/usr/bin/env python3
"""
Script to check if all required dependencies are available and their versions
"""

import sys
import subprocess
import pkg_resources
from packaging import version

def check_package(package_name, required_version=None):
    """Check if a package is installed and optionally verify version"""
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        print(f"✅ {package_name}: {installed_version}")
        
        if required_version:
            if version.parse(installed_version) >= version.parse(required_version):
                print(f"   Version requirement satisfied (>= {required_version})")
            else:
                print(f"   ⚠️  Version requirement NOT satisfied (>= {required_version})")
                return False
        return True
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"❌ {package_name}: Error checking - {e}")
        return False

def main():
    print("🔍 Checking Python dependencies for AI Psychologist...")
    print("=" * 60)
    
    # Core dependencies
    dependencies = [
        ("fastapi", "0.104.1"),
        ("uvicorn", "0.24.0"),
        ("google-genai", "0.8.0"),
        ("fastrtc", None),  # Check what version is actually available
        ("aiortc", "1.6.0"),
        ("numpy", "1.26.0"),
        ("scipy", "1.11.0"),
        ("av", "10.0.0"),
        ("python-dotenv", "1.0.0"),
        ("pydantic", "2.4.0"),
        ("httpx", "0.28.0"),
        ("websockets", "13.0"),
    ]
    
    all_good = True
    for package, min_version in dependencies:
        if not check_package(package, min_version):
            all_good = False
    
    print("\n" + "=" * 60)
    
    # Check FastRTC specifically
    print("\n🎯 FastRTC Version Check:")
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import fastrtc; print(f'FastRTC version: {fastrtc.__version__}')"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"❌ FastRTC import failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error checking FastRTC: {e}")
    
    # Check available FastRTC versions
    print("\n🔍 Available FastRTC versions:")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "fastrtc"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            # Fallback method
            result = subprocess.run([sys.executable, "-c", 
                                   "import subprocess; print(subprocess.check_output(['pip', 'install', 'fastrtc=='], text=True))"], 
                                  capture_output=True, text=True)
            if "available versions" in result.stderr:
                print(result.stderr)
    except Exception as e:
        print(f"Could not check available versions: {e}")
    
    if all_good:
        print("\n✅ All dependencies are properly installed!")
    else:
        print("\n⚠️  Some dependencies need attention. Check the output above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
