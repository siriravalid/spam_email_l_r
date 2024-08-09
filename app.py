import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']] 
df.columns = ['Label', 'Email']
df['Label'] = df['Label'].map({'spam': 'spam', 'ham': 'not spam'})

X = df['Email']
y = df['Label']
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Spam Email Classifier(logistic regression)-Siri")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

email_input = st.text_area("Enter the email content:")
if st.button("Classify"):
    if email_input:
        input_vector = vectorizer.transform([email_input])
        prediction = model.predict(input_vector)[0]
        st.write(f"The email is classified as: **{prediction}**")
    else:
        st.write("Please enter some email content.")
