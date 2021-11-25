from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Assume we have a dataset that is labeled by topic (technology, education etc.) then we can label the data and compare sentiment
# If we are able to split the data by date then we could show the trend over time?
# Quick test of simplest sentiment analyser tool using example categorised text data from bbc articles
if __name__ == '__main__':
    nltk.downloader.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv('bbc-text.csv')

    label_sentiment_dic = {}
    for i in range(0, df.__len__()):
        category = df['category'][i]
        compound_score = sia.polarity_scores(df['text'][i])['compound']
        if (label_sentiment_dic.get(category) is None):
            label_sentiment_dic[category] = compound_score
        else:
            label_sentiment_dic[category] += compound_score

    for k in label_sentiment_dic.keys():
        label_sentiment_dic[k] = round(label_sentiment_dic[k]/df.__len__(), 2)

    print(label_sentiment_dic)

