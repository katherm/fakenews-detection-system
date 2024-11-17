import streamlit as st
import pickle
import re
import pandas as pd
import string
import plotly.graph_objects as go
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import os

# Load models and vectorizer
with open(r'C:\Users\alfre\OneDrive\Desktop\switfathon2\jupyter\model_LR.pkl', 'rb') as file:
    LR = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\switfathon2\jupyter\model_DT.pkl', 'rb') as file:
    DT = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\switfathon2\jupyter\model_RF.pkl', 'rb') as file:
    RF = pickle.load(file)

with open(r'C:\Users\alfre\OneDrive\Desktop\switfathon2\jupyter\vectorizer.pkl', 'rb') as file:
    vectorization = pickle.load(file)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to predict news type using models
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return pred_LR[0], pred_DT[0], pred_RF[0]

# Function to format the output label
def output_label(n):
    return "Real News" if n == 1 else "Fake News"

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Sorry, there was an error with the speech recognition service.")

# Streamlit interface
st.title('📰 Fake News Classification App')
st.markdown("""
Welcome to the Fake News Classification App. This tool uses three different machine learning models 
(Logistic Regression, Decision Tree, and Random Forest) to predict whether a given news content is **Fake** or **Real**. 
Simply enter the news text below and click "Predict" to see the results. You can also use voice input to add news content.
""")

# Input section
st.subheader("Enter News Content")
sentence = st.text_area("Type or paste the news content here:", "", height=200)
examples = st.selectbox("Or select an example:", ["", "Breaking: Major event shocks the world...", "Expert claims new diet trend..."])

# Use the selected example if available
if examples:
    sentence = examples

# Voice input button
if st.button("Record Voice Input 🎤"):
    sentence = recognize_speech()

# Prediction button
predict_btt = st.button("Predict 🧠")

# Prediction and output
if predict_btt:
    if sentence:
        with st.spinner('Analyzing...'):
            pred_LR, pred_DT, pred_RF = manual_testing(sentence)
        
        # Display predictions in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression", output_label(pred_LR), "📊")
        col2.metric("Decision Tree", output_label(pred_DT), "🌳")
        col3.metric("Random Forest", output_label(pred_RF), "🌲")

        # Visualize predictions using Plotly
        fig = go.Figure(data=[
            go.Bar(name='Logistic Regression', x=['Prediction'], y=[pred_LR], marker_color='blue'),
            go.Bar(name='Decision Tree', x=['Prediction'], y=[pred_DT], marker_color='green'),
            go.Bar(name='Random Forest', x=['Prediction'], y=[pred_RF], marker_color='orange')
        ])
        fig.update_layout(
            title="Model Predictions",
            xaxis_title="Models",
            yaxis_title="Prediction (0 = Fake, 1 = Real)",
            barmode='group'
        )
        st.plotly_chart(fig)

        # Convert the result to speech and play it
        result_text = f"Logistic Regression predicts {output_label(pred_LR)}, Decision Tree predicts {output_label(pred_DT)}, and Random Forest predicts {output_label(pred_RF)}."
        audio_bytes = text_to_speech(result_text)
        st.audio(audio_bytes, format='audio/mp3')

        st.success("Prediction complete! See results above.")
    else:
        st.warning("Please enter news content for prediction.")





