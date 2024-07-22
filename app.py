#app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
import json
import random
import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

app = Flask(__name__,static_url_path='/static')

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

with open("intents.json") as file:
    data = json.load(file)

model_data = torch.load("data.pth")
words = model_data['words']
labels = model_data['labels']

input_size = len(words)
hidden_size = 8
output_size = len(labels)

model = Net(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag).reshape(1, -1)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    print("Login page accessed")
    return render_template("login.html")

@app.route("/register")
def register():
    print("Register page accessed")
    return render_template("register.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json["message"]
    
    results = model(torch.from_numpy(bag_of_words(user_input, words)).float())
    results_index = np.argmax(results.detach().numpy())
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]
    response = random.choice(responses)
   
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)