import sys
import subprocess

# List of packages to install
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'joblib']

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
print("All packages installed successfully!")