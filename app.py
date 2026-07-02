import streamlit as st
import pickle
import matplotlib.pyplot as plt

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("Sentiment Analysis")

text = st.text_area("Enter text")

if st.button("Predict"):
    if text:
        data = vectorizer.transform([text])
        result = model.predict(data)[0]

        st.write("Sentiment:", result)

        labels = ["positive", "negative", "neutral"]
        values = [0, 0, 0]

        if result == "positive":
            values[0] = 1
        elif result == "negative":
            values[1] = 1
        else:
            values[2] = 1

        plt.bar(labels, values)
        st.pyplot(plt)