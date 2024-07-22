#model_train.py

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

def load_data(filepath):
    with open(filepath) as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    return training, output, words, labels

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

def train_model(model, training_data, output_data, epochs=1000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(training_data).float())
        loss = criterion(outputs, torch.from_numpy(np.argmax(output_data, axis=1)))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

def main():
    data = load_data("intents.json")
    training_data, output_data, words, labels = preprocess_data(data)
    
    input_size = len(training_data[0])
    hidden_size = 8
    output_size = len(output_data[0])
    
    model = Net(input_size, hidden_size, output_size)
    train_model(model, training_data, output_data)
    
    # Save the words and labels to use them later
    torch.save({'words': words, 'labels': labels}, "data.pth")
    print("Data saved to data.pth")

if __name__ == "__main__":
    main()