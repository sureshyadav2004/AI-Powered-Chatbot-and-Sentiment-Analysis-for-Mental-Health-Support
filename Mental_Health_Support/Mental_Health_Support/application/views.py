from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile

# Create your views here.

def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')
#MACHINE learning
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json

with open('static/Dataset/intents.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data['intents'])
dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
df = pd.DataFrame.from_dict(dic)
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()
vacab_size = len(tokenizer.word_index)
print('number of unique words = ', vacab_size)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')
print('X shape = ', X.shape)

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])
print('y shape = ', y.shape)
print('num of classes = ', len(np.unique(y)))
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
MODEL_PATH = 'static/model/model.h5'
import os
# Check if model exists
if os.path.exists(MODEL_PATH):
    # Load the pre-trained model
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    # Build and train the model if it does not exist
    print("Model not found. Training a new model.")

    model = Sequential()
    model.add(Input(shape=(X.shape[1])))
    model.add(Embedding(input_dim=vacab_size+1, output_dim=100, mask_zero=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32))
    model.add(LayerNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(len(np.unique(y)), activation="softmax"))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    # Train the model
    model_history = model.fit(
        x=X,
        y=y,
        batch_size=10,
        epochs=50,
        callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)]
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

import re
import random
from tensorflow.keras.models import load_model
def generate_answer(pattern): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
    model = load_model('static/model/model.h5')
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        bot_response = generate_answer(user_message)
        return JsonResponse({'response': bot_response})
    return render(request, 'chatbot.html')