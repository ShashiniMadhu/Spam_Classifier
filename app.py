import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os


# Download NLTK data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    return PorterStemmer()


ps = setup_nltk()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    text = [i for i in text if i.isalnum()]

    # Remove stopwords
    text = [i for i in text if i not in stopwords.words('english')]

    # Stem words
    text = [ps.stem(i) for i in text]

    return " ".join(text)


# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
    else:
        st.warning("Please enter a message")