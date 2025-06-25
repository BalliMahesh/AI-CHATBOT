pip install nltk
import nltk
import random
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Download necessary data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
True
# Tokenize and prepare the data
lemmer = WordNetLemmatizer()
sent_tokens = nltk.sent_tokenize(CORPUS.lower())

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text))
 #Greeting responses
greeting_inputs = ("hello", "hi", "hey", "greetings")
greeting_responses = ["Hello!", "Hi there!", "Hey!", "Greetings!"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
# Generate chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    temp_tokens = sent_tokens + [user_input]
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('english'))
    tfidf = vectorizer.fit_transform(temp_tokens)
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = similarity_scores.argsort()[0][-1]
    flat = similarity_scores.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        return "I'm sorry, I don't understand that."
    else:
        return sent_tokens[idx]
def chat():
    print("AI Bot: Hello! Ask me anything or type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("AI Bot: Goodbye!")
            break
        elif greet(user_input):
            print("AI Bot:", greet(user_input))
        else:
            print("AI Bot:", chatbot_response(user_input))
# Run chatbot
if __name__ == "__main__":
    chat()
