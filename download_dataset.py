
from roboflow import Roboflow
import os
import shutil

# Initialize Roboflow with your actual API key
rf = Roboflow(api_key="---Give your generated API Key")  # Replace with your actual API key

# First, let's see what workspaces you have access to
print("🔍 Available workspaces:")
try:
    workspaces = rf.list_workspaces()
    for workspace in workspaces:
        print(f"  - {workspace}")
except Exception as e:
    print(f"Error listing workspaces: {e}")

# Your workspace and project details from the URL
workspace_name = "appledetectionwithyolov8"
project_name = "tomato-leaf-dissease-detection"
version_number = 7  # From the URL

# Download dataset
try:
    print(f"🔍 Connecting to workspace: {workspace_name}")
    print(f"🔍 Project: {project_name}")
    print(f"🔍 Version: {version_number}")
    
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version_number).download("yolov8")  # Using YOLOv8 format
    
    print(f"✅ Successfully downloaded dataset!")
    successful_project_name = project_name
    
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("💡 Make sure your API key has access to this public dataset")
    exit()

print(f"✅ Dataset downloaded to: {dataset.location}")
print(f"📊 Classes: {dataset.classes}")

# Get all images from the dataset
image_dir = os.path.join(dataset.location, "train")  # or "valid"/"test" depending on which split you want
all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"📸 Total images found: {len(all_images)}")

# Pick first 20 images
selected_images = all_images[:20]

# Create sample directory and copy images
os.makedirs("sample_20", exist_ok=True)
for img in selected_images:
    src_path = os.path.join(image_dir, img)
    dst_path = os.path.join("sample_20", img)
    shutil.copy(src_path, dst_path)
    print(f"Copied: {img}")

print(f"✅ {len(selected_images)} images downloaded and saved in 'sample_20'")
print(f"📁 Sample images location: {os.path.abspath('sample_20')}")

# Optional: Display dataset structure
print(f"\n📂 Dataset structure:")
for root, dirs, files in os.walk(dataset.location):
    level = root.replace(dataset.location, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files in each directory
        print(f"{subindent}{file}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")
