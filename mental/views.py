from django.shortcuts import render
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    gender = request.GET.get('gender')

    try:
        val1 = 0 if gender == 'male' else float(request.GET.get('n1', 0))
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = int(request.GET['n8'])
    except ValueError:
        return render(request, 'predict.html', {"result2": "Invalid Input"})

    # ✅ Use correct path to load pred.csv
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'pred.csv')

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return render(request, 'predict.html', {"result2": "Model data not found on server."})

    # ❌ You used "data" instead of "df"
    X = df.drop("Outcome", axis=1)
    Y = df["Outcome"]

    # Train model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Predict
    pred = model.predict([[val1, val2, val3, val5, val6, val7, val8]])
    result1 = "Positive" if pred[0] == 1 else "Negative"

    context = {
        "result": result1,
        "age": val8,
        "gender": gender,
    }

    return render(request, 'result.html', context)
