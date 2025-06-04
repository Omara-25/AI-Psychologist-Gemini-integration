#!/usr/bin/env python3
"""
AI Psychologist with Gemini Voice Integration - Setup Script

This script helps set up the development environment and verify configuration.
"""

import os
import sys
import subprocess
import secrets
import platform
from pathlib import Path

def print_banner():
    """Print the setup banner"""
    print("=" * 60)
    print("🧠 AI Psychologist with Gemini Voice Integration")
    print("   Setup and Configuration Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_system_dependencies():
    """Check for required system dependencies"""
    print("\n🔍 Checking system dependencies...")
    
    dependencies = {
        "git": "Git version control",
        "curl": "HTTP client for testing",
    }
    
    missing_deps = []
    
    for cmd, description in dependencies.items():
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}: {description}")
            else:
                missing_deps.append(cmd)
        except FileNotFoundError:
            missing_deps.append(cmd)
            print(f"❌ {cmd}: {description} - NOT FOUND")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        if platform.system() == "Darwin":  # macOS
            print("   Install with: brew install " + " ".join(missing_deps))
        elif platform.system() == "Linux":
            print("   Install with: sudo apt-get install " + " ".join(missing_deps))
        return False
    
    return True

def create_virtual_environment():
    """Create and setup virtual environment"""
    print("\n🐍 Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Determine pip path based on OS
    if platform.system() == "Windows":
        pip_path = Path("venv/Scripts/pip")
    else:
        pip_path = Path("venv/bin/pip")
    
    if not pip_path.exists():
        print("❌ Virtual environment not found. Please run create_virtual_environment first.")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template"""
    print("\n⚙️  Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example template not found")
        return False
    
    # Read template
    with open(env_example, 'r') as f:
        env_content = f.read()
    
    # Generate secret key
    secret_key = secrets.token_urlsafe(32)
    env_content = env_content.replace("your_secret_key_here", secret_key)
    
    # Generate JWT secret
    jwt_secret = secrets.token_urlsafe(32)
    env_content = env_content.replace("your_jwt_secret_here", jwt_secret)
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created with generated secrets")
    print("⚠️  Please edit .env and add your GEMINI_API_KEY")
    return True

def verify_gemini_api_key():
    """Check if Gemini API key is configured"""
    print("\n🔑 Checking Gemini API key configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    if "your_gemini_api_key_here" in env_content:
        print("⚠️  Please add your Gemini API key to .env file")
        print("   Get your key at: https://ai.google.dev/gemini-api/docs/api-key")
        return False
    
    # Check if API key line exists and is not empty
    for line in env_content.split('\n'):
        if line.startswith('GEMINI_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
            if api_key and api_key != "your_gemini_api_key_here":
                print("✅ Gemini API key is configured")
                return True
    
    print("⚠️  Gemini API key not found in .env file")
    return False

def test_application():
    """Test basic application functionality"""
    print("\n🧪 Testing application...")
    
    # Determine python path
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python")
    else:
        python_path = Path("venv/bin/python")
    
    if not python_path.exists():
        print("❌ Virtual environment not found")
        return False
    
    try:
        # Test basic import
        result = subprocess.run([
            str(python_path), "-c", 
            "from main import app; print('✅ Application imports successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Import test timed out")
        return False
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def run_quick_tests():
    """Run quick functionality tests"""
    print("\n🔬 Running quick tests...")
    
    # Determine python path
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python")
    else:
        python_path = Path("venv/bin/python")
    
    try:
        result = subprocess.run([
            str(python_path), "test_main.py"
        ], capture_output=True, text=True, timeout=60)
        
        if "✅" in result.stdout:
            print("✅ Basic tests passed")
            return True
        else:
            print("⚠️  Some tests may have failed")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")
        return False

def create_development_scripts():
    """Create helpful development scripts"""
    print("\n📝 Creating development scripts...")
    
    scripts = {
        "run_dev.py": '''#!/usr/bin/env python3
"""Development server runner"""
import subprocess
import sys
from pathlib import Path

def main():
    if Path("venv").exists():
        if sys.platform == "win32":
            python_path = "venv/Scripts/python"
        else:
            python_path = "venv/bin/python"
        subprocess.run([python_path, "main.py"])
    else:
        print("Virtual environment not found. Run setup.py first.")

if __name__ == "__main__":
    main()
''',
        "run_tests.py": '''#!/usr/bin/env python3
"""Test runner"""
import subprocess
import sys
from pathlib import Path

def main():
    if Path("venv").exists():
        if sys.platform == "win32":
            python_path = "venv/Scripts/python"
            pytest_path = "venv/Scripts/pytest"
        else:
            python_path = "venv/bin/python"
            pytest_path = "venv/bin/pytest"
        
        if Path(pytest_path).exists():
            subprocess.run([pytest_path, "test_main.py", "-v"])
        else:
            subprocess.run([python_path, "-m", "pytest", "test_main.py", "-v"])
    else:
        print("Virtual environment not found. Run setup.py first.")

if __name__ == "__main__":
    main()
'''
    }
    
    for script_name, script_content in scripts.items():
        script_path = Path(script_name)
        if not script_path.exists():
            with open(script_path, 'w') as f:
                f.write(script_content)
            # Make executable on Unix systems
            if platform.system() != "Windows":
                os.chmod(script_path, 0o755)
            print(f"✅ Created {script_name}")
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your Gemini API key:")
    print("   GEMINI_API_KEY=your_actual_api_key_here")
    print()
    print("2. Start the development server:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python main.py")
    else:
        print("   ./run_dev.py")
        print("   # or")
        print("   venv/bin/python main.py")
    print()
    print("3. Open your browser to: http://localhost:8000")
    print()
    print("4. For deployment to Railway:")
    print("   - Install Railway CLI: npm install -g @railway/cli")
    print("   - Follow the deployment guide in DEPLOYMENT_PLAN.md")
    print()
    print("📚 Documentation:")
    print("   - README.md - Full project documentation")
    print("   - DEPLOYMENT_PLAN.md - Deployment guide")
    print("   - .env.example - Environment variable reference")
    print()
    print("🧪 Testing:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\pytest test_main.py -v")
    else:
        print("   ./run_tests.py")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    success = True
    
    # Check requirements
    if not check_python_version():
        success = False
    
    if not check_system_dependencies():
        success = False
    
    if not success:
        print("\n❌ Please fix the above issues before continuing.")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating environment file", create_env_file),
        ("Creating development scripts", create_development_scripts),
        ("Testing application", test_application),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed")
            success = False
            break
    
    # Optional steps that don't fail the setup
    verify_gemini_api_key()
    run_quick_tests()
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
