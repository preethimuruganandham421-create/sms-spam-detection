import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
data = pd.read_csv(
       "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(data.head())
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})
print(data["label"].value_counts())
X = data["message"]
y = data["label_num"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
accuracy = model.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        message = request.form["message"]
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)

        if prediction[0] == 1:
            result = "🚨 SPAM"
        else:
            result = "✅ NOT SPAM"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
  
