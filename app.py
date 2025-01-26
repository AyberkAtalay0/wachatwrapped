from typing import Counter
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import traceback 

emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F]"  # Smileys
    "|[\U0001F300-\U0001F5FF]"  # Symbols & Pictographs
    "|[\U0001F680-\U0001F6FF]"  # Transport & Map Symbols
    "|[\U0001F700-\U0001F77F]"  # Alchemical Symbols
    "|[\U0001F780-\U0001F7FF]"  # Geometric Shapes Extended
    "|[\U0001F800-\U0001F8FF]"  # Supplemental Arrows-C
    "|[\U0001F900-\U0001F9FF]"  # Supplemental Symbols and Pictographs
    "|[\U0001FA00-\U0001FA6F]"  # Chess Symbols
    "|[\U0001FA70-\U0001FAFF]"  # Symbols and Pictographs Extended-A
    "|[\U00002702-\U000027B0]"  # Dingbats
    "|[\U000024C2-\U0001F251]"  # Enclosed Characters
    "|\u200d"  # Zero Width Joiner
    "|\u2640"  # Female Sign
    "|\u2642"  # Male Sign
    "|\u2600-\u2B55"  # Miscellaneous Symbols
    "|\u23cf"  # Eject Button
    "|\u23e9-\u23ef"  # AV Symbols
    "|\u231a-\u231b"  # Watch
    "|\ufe0f"  # Variation Selectors
    "|\u3030"  # Wavy Dash
    "|\u2122"  # Trademark
    "|\u23f0"  # Alarm Clock
    "|\u23f3"  # Hourglass
    "|\u24c2"  # Circled Letter
    "|\u25aa-\u25ab"  # Black and White Squares
    "|\u25b6"  # Play Button
    "|\u25c0"  # Reverse Play Button
    "|\u25fb-\u25fe"  # White and Black Squares
    "|\u2600-\u26FF"  # Miscellaneous Symbols
    "|\u2B05-\u2B07"  # Arrows
    "|\u2934-\u2935"  # Arrows Extended
    "|\u2b1b-\u2b1c"  # Black Squares
    "|\u2b50"  # Star
    "|\u2b55"  # Circle
    "|\u303d"  # Part Alternation Mark
    "|\u3297"  # Circled Ideograph Congratulation
    "|\u3299"  # Circled Ideograph Secret
    "|[\U0001F1E6-\U0001F1FF]"  # Regional Indicator Symbols
    "]+", 
    flags=re.UNICODE,
)

def extract_emojis(text):
    global emoji_pattern
    modifiers_pattern = re.compile("[\U0001F3FB-\U0001F3FF]", flags=re.UNICODE)
    emojis = emoji_pattern.findall(text)
    return [emoji for emoji in emojis if not modifiers_pattern.match(emoji)]

def remove_emojis(text):
    global emoji_pattern
    return emoji_pattern.sub(r'', text)

def get_names(data):
    _names = []
    for i in range(len(data)):
        try:
            _name = data[i].split("-")[1].split(":")[0].strip()
            if _name not in _names and data[i].count(":") > 1 and data[i].count("-") > 0 and data[i].count(".") > 1 and data[i].index(".") < data[i].index(":") < data[i].index("-"): _names.append(_name)
        except: pass
    _names.sort()
    return _names

def get_messages_by_owner(data, owner):
    _messages = []
    for i in range(len(data)):
        _splitted_data = data[i].split(":")
        if owner in "".join(_splitted_data[:2]) and len(_splitted_data) > 1:
            _messages.append(_splitted_data[2].removesuffix("\n").strip())
    return _messages

def get_hours_by_owner(data, owner):
    _hours = []
    for i in range(len(data)):
        if owner in "".join(data[i].split(":")[:2]):
            _hours.append(float(data[i].split()[1].strip().split(":")[0]))
    return _hours

def get_sentiment_array(data, amplify_factor=1):
    analyzer = SentimentIntensityAnalyzer()
    _sentiments = []
    for _sentence in data:
        _score = analyzer.polarity_scores(remove_emojis(_sentence))
        amplified_score = _score['compound'] * amplify_factor
        amplified_score = max(min(amplified_score, 1), -1)
        _sentiments.append(amplified_score)
    return _sentiments

def get_used_emojis(data):
    _all_emojis = []
    for _message in data:
        emojis = extract_emojis(_message)
        _all_emojis.extend(emojis)
    return _all_emojis

def get_media_count(data):
    _media = 0
    for i in range(len(data)):
        if data[i].lower().startswith("<") and data[i].endswith(">"): _media += 1
    return _media

def get_urls(data):
    _urls = []
    for i in range(len(data)):
        _splitted_data = data[i].split()
        for _data in _splitted_data:
            if _data.startswith("http"): _urls.append(_data)
    return _urls

def get_most_used_words(data):
    words = []
    for _message in data:
        if _message.startswith("<") and _message.endswith(">"): return
        _message = remove_emojis(_message).lower()
        words.extend(_message.split())
    
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(5)
    return most_common_words

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app)
executor = ThreadPoolExecutor()

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Active"}), 200

@app.route("/wrapped", methods=["POST"])
@limiter.limit("5 per minute")
def wrapped():
    result = {}

    try:
        data = request.get_json()
        chat = data.get("chat")
        raw = chat.split("\n")[1:]

        names = get_names(raw)
        for name in names:
            result[name] = {}

            hours = get_hours_by_owner(raw, name)
            most_active_hour = str(int(max(set(hours), key=hours.count)))
            if len(most_active_hour) < 2: most_active_hour = "0" + most_active_hour + ":00"
            result[name]["most_active_hour"] = most_active_hour

            messages = get_messages_by_owner(raw, name)
            result[name]["messages"] = len(messages)
            characters = sum([len(message) for message in messages])
            result[name]["characters"] = characters
            result[name]["avg_msg_length"] = characters / len(messages)

            used_emojis = get_used_emojis(messages)
            emoji_frequencies = list({emoji: used_emojis.count(emoji)/len(used_emojis) for emoji in used_emojis}.items())
            emoji_frequencies.sort(key=lambda x: x[1], reverse=True)
            result[name]["emojis"] = emoji_frequencies

            media_count = get_media_count(messages)
            result[name]["media_count"] = media_count

            urls = get_urls(messages)
            url_count = len(urls)
            result[name]["url_count"] = url_count

            sentiments = get_sentiment_array(messages)
            valuable_sentiments = [i for i in sentiments if i != 0]
            avg_sentiment = np.mean(valuable_sentiments)
            result[name]["avg_sentiment"] = avg_sentiment
            std_sentiment = np.std(valuable_sentiments)
            result[name]["std_sentiment"] = std_sentiment
            balance_sentiment = sum(valuable_sentiments)
            result[name]["balance_sentiment"] = balance_sentiment

            most_used_words = get_most_used_words(messages)
            result[name]["most_used_words"] = most_used_words

        return jsonify({"result": result}), 200
    except:
        traceback.print_exc() 
        return jsonify({"error": str(1)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
