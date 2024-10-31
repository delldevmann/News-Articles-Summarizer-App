from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from deep_translator import GoogleTranslator
import textwrap
import re

# Load models and utilities
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Set up pipelines with device specification (0 = GPU, -1 = CPU)
sentiment_analyzer = pipeline("sentiment-analysis", device=0)  # Adjust device based on availability
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def translate_text(text, target_language="en"):
    """Translate text using deep-translator's GoogleTranslator."""
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(text)

def text_summarizer(text, max_length=125, min_length=50, length_penalty=2.0, num_beams=4, 
                    target_language=None, detailed=False):
    # Translate if needed
    if target_language:
        text = translate_text(text, target_language=target_language)
    
    # Summarize text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, 
                                 length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    formatted_summary = re.sub(r'\s+', ' ', formatted_summary)

    # Sentiment analysis
    sentiment = sentiment_analyzer(text)[0]
    
    # Topic detection
    labels = ["business", "technology", "politics", "sports", "entertainment", "science"]
    topic = topic_classifier(text, candidate_labels=labels)
    main_topic = topic['labels'][0]  # Top detected topic

    # Keyword highlighting (simple approach with keywords)
    keywords = ["breaking", "important", "urgent", "exclusive"]
    for keyword in keywords:
        formatted_summary = re.sub(f"\\b{keyword}\\b", f"*{keyword.upper()}*", formatted_summary)

    # Prepare final output
    result = {
        "summary": formatted_summary,
        "sentiment": sentiment,
        "topic": main_topic,
    }

    return result

# Example usage
if __name__ == "__main__":
    text = "Your long article or text goes here..."
    result = text_summarizer(text, target_language="es", detailed=True)
    print(result)
