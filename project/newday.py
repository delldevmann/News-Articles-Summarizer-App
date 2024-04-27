import os
from main3 import start
from main5 import start_new
import nltk
from pathlib import Path
nltk.download('punkt')
nltk.download('stopwords')
csv_folder = Path(__file__).resolve().parent  # Get the directory of the script
files = [
    csv_folder/'india.csv',
    csv_folder/'world.csv',
    csv_folder/'business.csv',
    csv_folder/'tech.csv',
    csv_folder/'sports.csv'
         ]

for filepath in files:
    # Check if the file exists
    if os.path.exists(filepath):
        # Open the file in binary write mode
        with open(filepath, "wb") as f:
            # Truncate the file
            f.truncate()
        print(f"File '{filepath}' truncated successfully.")
    else:
        print(f"File '{filepath}' does not exist.")

start_new()