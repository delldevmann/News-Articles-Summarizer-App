import os
import nltk
from pathlib import Path
from main3 import start
from main5 import start_new

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Get the directory of the script
csv_folder = Path(__file__).resolve().parent  

# Define file paths for various CSVs
files = [
    csv_folder/'india.csv',
    csv_folder/'world.csv',
    csv_folder/'business.csv',
    csv_folder/'tech.csv',
    csv_folder/'sports.csv'
]

# Truncate files if they exist
for filepath in files:
    if os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.truncate()
        print(f"File '{filepath}' truncated successfully.")
    else:
        print(f"File '{filepath}' does not exist.")

# Run the required startup functions
start_new()
