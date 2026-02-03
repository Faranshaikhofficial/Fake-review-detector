# 🔍 Fake Review Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-green.svg)

## 📌 Project Overview
The **Fake Review Detection System** is a Machine Learning powered web application designed to identify deceptive or non-genuine product reviews. Unlike traditional methods that only look at keywords, this system uses a novel **"Behavioral DNA"** approach—analyzing writing patterns, punctuation intensity, and vocabulary diversity to flag fake content with **93.5% accuracy**.

## 🚀 Key Features
- **🧬 Behavioral DNA Analysis**: Detects anomalies like "shouting" (all caps), excessive punctuation (!!), and repetitive vocabulary.
- **🧠 Machine Learning Core**: Uses Logistic Regression with TF-IDF vectorization trained on a balanced dataset.
- **📊 Interactive Dashboard**: A premium Streamlit UI with Dark Mode for real-time analysis.
- **📂 Bulk Analysis**: Upload CSV files (e.g., from Amazon/Flipkart) to risk-score hundreds of reviews at once.
- **💡 Explainable AI**: Visualizes *why* a review is marked fake using SHAP values and highlighting suspicious words.

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Plotly (for interactive charts)
- **Backend**: Python 3.9+
- **Machine Learning**: Scikit-learn (Logistic Regression, TF-IDF)
- **Data Processing**: Pandas, NumPy, Regex

## ⚙️ Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Faranshaikhofficial/Fake-review-detector.git
   cd Fake-review-detectior
   ```

2. **Install Dependencies**
   (It is recommended to use a virtual environment)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the App**
   Open your browser and go to `http://localhost:8501`

## 📸 Screenshots

### Home Dashboard
_Real-time analytics and model performance metrics._
<img width="1819" height="844" alt="image" src="https://github.com/user-attachments/assets/63b4ceb3-18ea-45a8-af31-9aef24a82588" />


### Behavioral Analytics
_Radar charts comparing the "DNA" of real vs. fake reviews._
<img width="1744" height="723" alt="image" src="https://github.com/user-attachments/assets/312186cc-04cc-455e-b214-b74c23743c64" />

## 🤝 Contribution
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
