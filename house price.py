# Welcome to GitHub Desktop!
from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample training dataset
data = {
    'Total_Area': [2000, 1500, 1800, 2200, 1600],
    'Living_Room': [400, 300, 350, 420, 310],
    'Hall': [300, 250, 270, 310, 280],
    'Bedroom': [1, 3, 4, 5, 7],
    'Kitchen': [200, 180, 190, 210, 185],
    'Bathroom': [3, 5, 7, 8, 9],
    'Price': [400000, 300000, 350000, 450000, 320000]
}
df = pd.DataFrame(data)

# Train model
X = df[['Total_Area', 'Living_Room', 'Hall', 'Bedroom', 'Kitchen', 'Bathroom']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# HTML template embedded directly in Python
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }
        .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 8px; }
        input[type="number"] { width: 100%; padding: 10px; margin: 5px 0 15px; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #28a745; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .result { margin-top: 20px; font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏡 House Price Predictor</h2>
        <form action="/predict" method="post">
            <label>Total Area (sq ft):</label>
            <input type="number" name="total_area" required>
            <label>Living Room Area (sq ft):</label>
            <input type="number" name="living_room" required>
            <label>Hall Area (sq ft):</label>
            <input type="number" name="hall" required>
            <label>Bedroom Count:</label>
            <input type="number" name="bedroom" required>
            <label>Kitchen Area (sq ft):</label>
            <input type="number" name="kitchen" required>
            <label>Bathroom Count:</label>
            <input type="number" name="bathroom" required>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction %}
            <div class="result">
                💰 Estimated Price: <strong>{{ prediction }}</strong><br>
                🛏️ Total Rooms (Bedroom + Bathroom): <strong>{{ room_count }}</strong>
            </div>
        {% elif error %}
            <div class="result" style="color: red;">
                ⚠️ Error: {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_area = float(request.form['total_area'])
        living_room = float(request.form['living_room'])
        hall = float(request.form['hall'])
        bedroom = int(request.form['bedroom'])
        kitchen = float(request.form['kitchen'])
        bathroom = int(request.form['bathroom'])

        features = np.array([[total_area, living_room, hall, bedroom, kitchen, bathroom]])
        prediction = model.predict(features)[0]

        # ✅ Room count = Bedroom + Bathroom
        room_count = bedroom + bathroom

        return render_template_string(html_template,
                                      prediction=f"${prediction:,.2f}",
                                      room_count=room_count)
    except Exception as e:
        return render_template_string(html_template, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)