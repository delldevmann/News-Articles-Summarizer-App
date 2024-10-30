from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from googletrans import Translator
import textwrap
import re

# Load models and utilities
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
translator = Translator()
sentiment_analyzer = pipeline("sentiment-analysis")
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def text_summarizer(text, max_length=125, min_length=50, length_penalty=2.0, num_beams=4, 
                    target_language=None, detailed=False):
    # Translate if needed
    if target_language:
        text = translator.translate(text, dest=target_language).text
    
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
text = "Your long article or text goes here..."
result = text_summarizer(text, target_language="es", detailed=True)
print(result)
