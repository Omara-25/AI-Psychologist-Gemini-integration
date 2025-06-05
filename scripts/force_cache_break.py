#!/usr/bin/env python3
"""
Force Docker cache break by updating build timestamp
"""

import os
from datetime import datetime

def update_dockerfile_timestamp():
    """Update the BUILD_VERSION in Dockerfile to force cache break"""
    
    dockerfile_path = "Dockerfile"
    
    if not os.path.exists(dockerfile_path):
        print("❌ Dockerfile not found")
        return False
    
    # Read current Dockerfile
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    # Generate new timestamp
    new_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    # Update BUILD_VERSION
    if "BUILD_VERSION=" in content:
        # Replace existing BUILD_VERSION
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "BUILD_VERSION=" in line:
                lines[i] = f"    BUILD_VERSION={new_timestamp}"
                break
        content = '\n'.join(lines)
    else:
        # Add BUILD_VERSION if not present
        content = content.replace(
            "ENV PYTHONDONTWRITEBYTECODE=1",
            f"ENV PYTHONDONTWRITEBYTECODE=1 \\\n    BUILD_VERSION={new_timestamp}"
        )
    
    # Write updated Dockerfile
    with open(dockerfile_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated BUILD_VERSION to {new_timestamp}")
    print("🔄 This will force Docker to rebuild from the pip install step")
    return True

def main():
    print("🔨 Force Docker Cache Break")
    print("=" * 30)
    
    if update_dockerfile_timestamp():
        print("\n📝 Next steps:")
        print("1. Commit and push the updated Dockerfile")
        print("2. Railway will rebuild without using cached layers")
        print("3. This ensures aiortc 1.6.0 is actually installed")
    else:
        print("\n❌ Failed to update Dockerfile")
    
    return 0

if __name__ == "__main__":
    exit(main())
