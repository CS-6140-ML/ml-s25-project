from textblob import TextBlob


def analyze_sentiment(text):
    """
    Analyze sentiment of the given text using TextBlob.

    Returns:
        tuple: (polarity, subjectivity)
            - polarity: float in range [-1, 1] (negative to positive sentiment)
            - subjectivity: float in range [0, 1] (objective to subjective)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def batch_sentiment_analysis(texts):
    """
    Analyze sentiment for a list of texts.

    Returns:
        list of tuples: (polarity, subjectivity) for each text.
    """
    return [analyze_sentiment(text) for text in texts]


if __name__ == "__main__":
    sample_texts = [
        "I love this restaurant!",
        "The service was terrible."
    ]
    sentiments = batch_sentiment_analysis(sample_texts)
    for text, sentiment in zip(sample_texts, sentiments):
        print(f"Text: {text}\nSentiment (polarity, subjectivity): {sentiment}\n")
