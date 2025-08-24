import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os


# Download NLTK data
@st.cache_resource
def setup_nltk():
    # For newer NLTK versions (3.9+)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to older punkt if punkt_tab fails
            nltk.download('punkt', quiet=True)

    # Also try the older punkt tokenizer as fallback
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

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


# Load models with error handling
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


tfidf, model = load_models()

# UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        try:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.header("SPAM")
            else:
                st.header("NOT SPAM")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a message")