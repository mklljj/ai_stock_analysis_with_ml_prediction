# 🤖 AI Stock Analyzer

An advanced stock analysis tool that combines technical indicators, multi-source sentiment analysis, and machine learning predictions to provide a comprehensive, AI-generated summary of a stock's potential.



## ✨ Core Features

* **Technical Analysis**: Automatically calculates key indicators like **SMA**, **MACD**, and **RSI** from real-time stock data.
* **Multi-Source Sentiment**: Aggregates and analyzes sentiment from three distinct sources:
    * **Financial News** (via Finnhub & NewsAPI)
    * **Social Media** (via Reddit)
* **Weighted Sentiment Score**: Combines sentiment from all sources into a single, weighted score for a more balanced perspective.
* **Machine Learning Predictions**: Trains four different regression models (**Linear Regression, Random Forest, LightGBM, XGBoost**) on historical data to predict the stock's price change over the next 25 minutes.
* **AI-Powered Summary**: Uses **Google's Gemini model** to synthesize all the technical, sentiment, and ML data into a concise, human-readable analysis and trading strategy.
* **Interactive UI**: A clean, responsive frontend with an interactive price chart (powered by Chart.js) to visualize historical data.

---

## 🛠️ Tech Stack

* **Backend**: Flask (Python)
* **Frontend**: HTML, CSS, JavaScript (no frameworks)
* **Data & ML**: Pandas, Scikit-learn, XGBoost, LightGBM
* **APIs**: Alpha Vantage, Finnhub, NewsAPI, Reddit, Google Gemini

---

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

* Python 3.8+
* Access to the required API keys (see configuration step).

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/mklljj/ai-stock-analyzer.git](https://github.com/mklljj/ai-stock-analyzer.git)
    cd ai-stock-analyzer
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure your API Keys:**
    Create a file named `.env` in the root of the project directory. This file stores your secret API keys. Add the following keys to it:

    ```env
    # Your secret key for authorizing frontend requests
    API_KEY="my-secret-stock-api-key-2024"

    # API Keys for data services
    ALPHA_VANTAGE_KEY="YOUR_ALPHA_VANTAGE_API_KEY"
    GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    NEWS_API_KEY="YOUR_NEWSAPI_KEY"
    FINNHUB_API_KEY="YOUR_FINNHUB_API_KEY"

    # Reddit API Credentials
    REDDIT_CLIENT_ID="YOUR_REDDIT_CLIENT_ID"
    REDDIT_CLIENT_SECRET="YOUR_REDDIT_CLIENT_SECRET"
    REDDIT_USER_AGENT="stock_analyzer_bot/1.0"
    ```

5.  **Run the Flask server:**
    ```sh
    python stock_server.py
    ```

6.  **Open the application:**
    Navigate to `http://localhost:8080` in your web browser.

---

## 📂 File Structure

```
/
├── stock_server.py         # Main Flask application, API routes
├── ml_predictor.py         # Machine learning model training and prediction
├── index.html              # Frontend UI structure
├── script.js               # Frontend logic, API calls, chart rendering
├── style.css               # All CSS styles for the frontend
├── requirements.txt        # Python dependencies
├── .gitignore              # Files and folders to ignore in git
└── README.md               # This file
```
*Note: The provided `config.py` file is currently not used by the application but can be integrated for more dynamic ML model configuration.*

---

## 📧 Contact

* **Bojian Zhou** - [neofastio@gmail.com](mailto:neofastio@gmail.com)
* **LinkedIn**: [https://www.linkedin.com/in/bojian-zhou-62391637b/](https://www.linkedin.com/in/bojian-zhou-62391637b/)
* **Project Link**: [https://github.com/mklljj/ai_stock_analysis_with_ml_prediction](https://github.com/mklljj/ai_stock_analysis_with_ml_prediction)
