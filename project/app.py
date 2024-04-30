import streamlit as st
import pandas as pd
from gtts import gTTS
import os
from pathlib import Path

vidno=0
st.set_page_config(layout="wide")
st.write(f"# News Articles")
# Define CSV file paths for each category
csv_folder = Path(__file__).resolve().parent  # Get the directory of the script
category_csv_files = {
    'India': csv_folder / 'india.csv',
    'World': csv_folder / 'world.csv',
    'Business': csv_folder / 'business.csv',
    'Technology': csv_folder / 'tech.csv',
    'Sports': csv_folder / 'sports.csv'
}

# Dropdown menu for category selection
selected_category = st.selectbox('Select Category', ['India', 'World', 'Business', 'Technology', 'Sports'])
st.write(f"## {selected_category}")
# Read the CSV file based on the selected category
csv_file = category_csv_files[selected_category]
df = pd.read_csv(csv_file)

# Display containers for each news article

for i in range(min(50, len(df))):
    article_title = df.iloc[i]['Article Title']
    article_summary = df.iloc[i]['Article Summary']
    article_link = df.iloc[i]['Article Link']
    article_image = df.iloc[i]['Article Image']
    
    # Check if all required fields are not empty and valid
    if all(isinstance(field, str) and field.strip() for field in [article_title, article_summary, article_link, article_image]):
        # Display article container
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("")
            st.write("")
            st.image(article_image, width=250)  # Display article image on the left side with specified height and width
            # Display "Read Full Article" button
            st.write(f"[Read Full Article]({article_link})")
        with col2:
            # Replace dollar signs with escape character before writing to Streamlit
            clean_title = article_title.replace('$', '\$')
            st.write(f"### {clean_title}")
            clean_summary = article_summary.replace('$', '\$')
            st.write(clean_summary)

            
            
            # Display "Convert to Audio" button
            convert_button_key = f"convert_button_{i}"
            if st.button("Convert to Audio", key=convert_button_key):
                # Convert summarized text to audio
                audio_filename = f"{vidno}_summary_audio.mp3"
                vidno=vidno+1
                tts = gTTS(article_summary, lang='en-uk')
                tts.save(audio_filename)
                st.audio(audio_filename, format='audio/mp3')

                # Remove audio file after playing
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    print(article_title, "- Audio file deleted")
            st.write("")
            st.write("")        
    else:
        print("One or more required fields are empty or invalid. Skipping article display.")

st.markdown("<p style='font-size: small; color: grey; text-align: center;'>A NLP project by <a href='https://github.com/akanksha1131'>Akanksha Kale</a>. Made for educational purposes only. Disclaimer: This project is intended for educational purposes only. Web scraping without proper authorization is not encourages or endorsed.</p>", unsafe_allow_html=True)    
