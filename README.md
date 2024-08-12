# Stock Market Predictor Dashboard

This Django-based web application is designed to predict stock prices using machine learning models. It allows users to load stock data, select specific indicators, and run a predictive model based on the selected inputs. The app also provides insights into recent news sentiment related to the chosen stock ticker.

## Features

- **Stock Data Loading**: Load historical stock data for specific tickers.
- **Feature Selection**: Choose various indicators like MACD, EMA, Bollinger Bands, and more to train the model.
- **News Sentiment Analysis**: Option to include sentiment analysis from recent news headlines about the selected stock.
- **Queue Management**: Handles multiple requests by placing them in a queue, showing the user their position.
- **Model Information Display**: Shows model type, loss, RMSE, R² values, and whether the model outperforms a naive approach.
- **Visualization**: Displays price, volume, and correlation charts with interactive zoom options.

## Getting Started

### Prerequisites

- Python 3.8+
- Django 3.2+
- Celery
- Redis
- yFinance
- PyTorch
- Transformers (Hugging Face)
- Selenium

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-market-predictor.git
   cd stock-market-predictor
   ```
2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Redis**:
   Install Redis on your machine or use a hosted Redis service.
   
4. **Run Migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Start Celery**:
   Ensure Celery is running for background task processing:
   ```bash
   celery -A website worker --loglevel=info
   ```
6. **Start the Django Development Server**:
   ```bash
   python manage.py runserver
   ```

## Configuration
- **Celery Configuration**: Modify the Celery settings in website/celery.py as per your environment.
- **Redis Configuration**: Ensure that the Redis settings in dashboard/utilies_helpers/utilies.py are pointing to the correct Redis server.

## Usage

1. Load Data:
- Select a stock ticker (AAPL, GOOG, NVDA).
- Choose the start and end dates.
- Configure model parameters like learning rate and factor.
- Optionally, check "Load Recent News" to include sentiment analysis.

2. Monitor Progress:
- Your position in the processing queue will be displayed.
- The progress bar will show the status of the data loading and model training.

3. View Results:
- Model results, including RMSE and R², will be displayed.
- Visualizations for prices, volume, and feature correlations will be available.

## Project Structure
- **views.py**: Contains the main logic for handling requests and loading data.
- **urls.py**: Defines URL patterns for the dashboard and data loading.
- **routing.py**: Configures WebSocket routing for real-time queue position updates.
- **fetch_stock_info**.py: Contains the Celery task for fetching stock data and calculating indicators.
- **recent_news.py**: Handles recent news scraping and sentiment analysis using a fine-tuned BERT model.
- **utilies.py**: Utility functions for task management, data validation, and Redis interaction.
- **basic.html**: The main template for the dashboard, featuring forms for data input and sections for displaying results.

## Future Enhancements
- **Unrestricted Historical News Data**: Plan to integrate more extensive historical news data.
- **Improved Headline Scraping**: Enhancing the efficiency of scraping news headlines.
- **Refining Sentiment Analysis**: Working on improving the accuracy of the sentiment analysis model.
By feeding the sentiment scores to the LSTM may yield better predictive power on closing prices.







   
