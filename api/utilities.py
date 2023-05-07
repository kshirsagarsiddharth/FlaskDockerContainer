import re
import pickle
from nltk.stem import WordNetLemmatizer

# Defining dictionary containing all emojis with their meanings.
emojis = {
    ":)": "smile",
    ":-)": "smile",
    ";d": "wink",
    ":-E": "vampire",
    ":(": "sad",
    ":-(": "sad",
    ":-<": "sad",
    ":P": "raspberry",
    ":O": "surprised",
    ":-@": "shocked",
    ":@": "shocked",
    ":-$": "confused",
    ":\\": "annoyed",
    ":#": "mute",
    ":X": "mute",
    ":^)": "smile",
    ":-&": "confused",
    "$_$": "greedy",
    "@@": "eyeroll",
    ":-!": "confused",
    ":-D": "smile",
    ":-0": "yell",
    "O.o": "confused",
    "<(-_-)>": "robot",
    "d[-_-]b": "dj",
    ":'-)": "sadsmile",
    ";)": "wink",
    ";-)": "wink",
    "O:-)": "angel",
    "O*-)": "angel",
    "(:-D": "gossip",
    "=^.^=": "cat",
}

## Defining set containing all stopwords in english.
stopwords = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "by",
    "can",
    "d",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "now",
    "o",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "own",
    "re",
    "s",
    "same",
    "she",
    "shes",
    "should",
    "shouldve",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "thatll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "y",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
]

lemmatizer = WordNetLemmatizer()


def preprocess(textdata):
    preprocessed_texts = []
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    user_pattern = r"@[^\s]+"
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"

    for tweet in textdata:
        # lower a individual tweet
        tweet = tweet.lower()
        # replace all the URL's with URL
        tweet = re.sub(url_pattern, " URL", tweet)

        # replace all the emojies
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        # replace all the usernames with USER
        tweet = re.sub(user_pattern, " USER", tweet)
        # replace all non alphabets
        tweet = re.sub(alpha_pattern, " ", tweet)
        # replace 3 or more consicitive letters with 2 letters
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        preprocessed_words = []
        for word in tweet.split():
            # check if the word is a stopword
            if len(word) > 1 and word not in stopwords:
                word = lemmatizer.lemmatize(word)
                preprocessed_words.append(word)

        preprocessed_texts.append(" ".join(preprocessed_words))

    return preprocessed_texts


with open("api/models/pipeline.pickle", "rb") as f:
    loaded_pipe = pickle.load(f)


def predict(model, text):
    pred_to_label = {0: "negative", 1: "positive"}
    preprocessed_text = preprocess(text)

    predictions = model.predict(preprocessed_text)

    data = []
    for t, pred in zip(text, predictions):
        data.append({"text": t, "predval": str(pred), "predlabel": pred_to_label[pred]})

    return data


def predict_pipeline(text):
    return predict(loaded_pipe, text)


if __name__ == "__main__":
    text = ["I hate twitter"]

    predictions = predict_pipeline(text)
    print(predictions)
