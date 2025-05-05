# 📈 Stock Price Prediction Model

A Streamlit-based web application that predicts future stock prices using LSTM (Long Short-Term Memory) neural networks. This project utilizes historical stock data from Yahoo Finance via the `yfinance` API and presents interactive visualizations and predictions to the user.

---

## 🚀 Features

- 📊 Fetch historical stock prices using Yahoo Finance
- 📉 Visualize Moving Averages (MA50, MA100, MA200)
- 🤖 Predict future prices using a trained LSTM deep learning model
- 📈 Compare predicted prices with actual prices
- 🌐 Interactive user interface using Streamlit

---

## 🛠️ Tech Stack

- Python 3.11+
- TensorFlow/Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- yfinance

---

## 🔧 Project Structure

```

📁 Stock Prediction Model/
│
├── main.py                     # Streamlit app to run the prediction
├── stocksanalysis.ipynb       # Jupyter notebook used to train and save the model
├── Stock Analysis Model.keras # Saved LSTM model
├── README.md                  # Project readme

````

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ranjangupta4590/Stock-Prediction-Model-Machine-Leaning.git
````

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate      # For Windows
# OR
source venv/bin/activate   # For macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install manually:

```bash
pip install numpy pandas yfinance matplotlib scikit-learn tensorflow streamlit
```

### 4. Train the Model (Optional)

If the `.keras` model is not already present, run the training script via notebook:

```bash
jupyter notebook stocksanalysis.ipynb
```

Make sure the last line in the notebook executes:

```python
model.save('F:/Stock Prediction Model/Stock Analysis Model.keras')
```

### 5. Run the Application

Make sure `main.py` is updated with the correct path to your `.keras` file.

Then, run:

```bash
streamlit run main.py
```

---

## 💡 How It Works

1. User enters a stock ticker symbol (e.g., `GOOG`)
2. The app fetches historical stock data from Yahoo Finance (2020 to 2025)
3. Visualizes:

   * Raw stock price
   * 50-day, 100-day, and 200-day moving averages
4. The LSTM model trained on the closing prices predicts future stock prices
5. The app compares and visualizes original vs predicted prices

---

## 🧠 Model Architecture

* 4 LSTM layers with ReLU activation and increasing units
* Dropout layers to prevent overfitting
* Final Dense layer to predict closing price
* Trained on 80% of historical closing prices, 20% used for testing
---

## 🤝 Contributing

Pull requests and suggestions are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is open-source and available under the MIT License.
