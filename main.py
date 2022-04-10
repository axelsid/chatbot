from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from constans import *
import numpy as np

import operator
import random
import pyttsx3
import wolframalpha

# speech Synthesis

engine = pyttsx3.init()


def speak(txt):
    engine.say(txt)
    engine.runAndWait()


# setup wolfram

client = wolframalpha.Client("7PHR7V-8XXVVLVPUT")

# ---------------------------------Create and train neural network-------------------------------------- #

tokenizer = Tokenizer(10000, lower=True)
binanizer = LabelBinarizer()

reactions = []
labels = []

reactions.extend(HIREACTIONS)
reactions.extend(HOWAREREACTIONS)
reactions.extend(NAMEREACTIONS)
reactions.extend(MATHREACTIONS)
reactions.extend(EXITREACTIONS)

labels.extend(HILABELS)
labels.extend(HOWARELABELS)
labels.extend(NAMELABELS)
labels.extend(MATHRLABELS)
labels.extend(EXITLABELS)

tokenizer.fit_on_texts(reactions)

reactions = pad_sequences(tokenizer.texts_to_sequences(reactions))
binanizer.fit(labels)
labels = binanizer.transform(labels)


def vectorizeSequences(sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))

    for i, seq in enumerate(sequence):
        results[i, seq] = 1

    return results


reactions = vectorizeSequences(reactions)

print(reactions)
print(labels)

network = Sequential()

network.add(Dense(64, activation="relu"))
network.add(Dropout(0.5))
network.add(Dense(64, activation="relu"))
network.add(Dropout(0.5))
network.add(Dense(5, activation="softmax"))

network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

network.fit(reactions, labels, epochs=500, verbose=0)

# -------------------------------main-----------------------------------#

while True:
    query = input("Ty: ")

    sequence = [query]

    sequence = tokenizer.texts_to_matrix(sequence)

    predictions = network.predict(sequence)[0]

    dictPred = {
        "greet": predictions[1],
        "hay": predictions[2],
        "math": predictions[3],
        "exit": predictions[0],
        "name": predictions[4]
    }

    ans = max(dictPred.items(), key=operator.itemgetter(1))[0]

    if ans == "greet":
        answer = random.choice(["cześć", "witaj", "hej"])

        print("Bot: " + answer)
        speak(answer)
    elif ans == "hay":
        answer = random.choice(["dobrze", "ok", "jakoś"])

        print("Bot: " + answer)

        speak(answer)
    elif ans == "math":
        print("Bot: Już liczę...")
        speak("Już liczę...")

        try:
            result = next(client.query(query).results).text
            print(f"Bot: {result}")
            speak(result)
        except:
            print("Bot: Niestety, nic nie znalazłam. Spróbuj inaczej")
            speak("Niestety, nic nie znalazłam. Spróbuj inaczej")

    elif ans == "exit":
        answer = random.choice(["dobrze", "oczywiście", "do widzenia", "żegnaj", "będę tu, jeśli będziesz mnie potrzebował"])

        print("Bot: " + answer)

        speak(answer)

        break
    elif ans == "name":
        answer = random.choice(["Jestem Sakura", "Mam na imię Sakura"])

        print("Bot: " + answer)

        speak(answer)
