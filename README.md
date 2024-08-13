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
   #### Installing Redis
   Redis is an in-memory data structure store that can be used as a message broker for Celery. Below are instructions for installing Redis on different platforms.

   ##### Install Redis on Ubuntu/Debian
   
   1. **Update your package list**:
      ```bash
      sudo apt-get update
      ```
   2. **Install Redis:**
      ```bash
      sudo apt-get install redis-server
      ```
   3. **Start and enable Redis to run on system boot:**
      ```bash
      sudo systemctl enable redis-server
      sudo systemctl start redis-server
      ```
   4. **Verify that Redis is running:**
      ```bash
      sudo systemctl status redis-server
      ```
      Ensure redis is operating on port 6379.

5. **Run Migrations**:
   ```bash
   python manage.py migrate
   ```

6. **Start Celery**:
   Ensure Celery is running for background task processing - navigate to the ```Stock_Market_Predictor_Web_App/website``` directory and use the following command:
   ```bash
   celery -A website worker --loglevel=info
   ```
7. **Installing and Setting Up Daphne:**
   Daphne is an HTTP, HTTP2, and WebSocket server for ASGI and is often used in Django projects that utilize Django Channels for handling WebSocket connections. Below are instructions for installing and configuring Daphne.

   To start your Django application with Daphne, navigate to the ```Stock_Market_Predictor_Web_App/website``` directory and use the following command:
   ```bash
   daphne -p 8001 website.asgi:application
   ```
   The website is set up to use port 8001 for websockets.
   
9. **Start the Django Development Server**:
   Navigate to the ```Stock_Market_Predictor_Web_App/website``` directory and use the following command:
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
```
website/
├── dashboard/
| ├── templates/
| | └── dashboard/
| | | └── basic.html - Contains the base html script to render the webpage.
| ├── pytorch_models/
| | ├── sentiment_model.py - Entity sentiment analysis model.
| | └── train_model.py - Trains the LSTM to predict stock prices.
| ├── utilities_helpers/
| | ├── tokeniszer_directory/ - (Your entity sentiment tokeniszer, refer to 'Downloading Entity-Sentiment Model Section')
| | ├── enitire_model_state_dict.pth (Your entity sentiment model, refer to 'Downloading Entity-Sentiment Model Section')
| | ├── fetch_stock_info.py - Contains the Celery task for fetching stock data.
| | ├── recent_news.py -  Handles recent news scraping and sentiment analysis using a fine-tuned BERT model.
| | └── utilities.py - Handles tasks once completed and containers helper functions for views.py such as validating data and turning get request to a dictionary.
│ ├── admin.py
│ ├── apps.py
│ ├── consumers.py - Websocket for the queue position counter.
│ ├── models.py
│ ├── routing.py - Configures WebSocket routing for real-time queue position updates.
│ ├── urls.py
│ └── views.py
├── website/
│ ├── asgi.py
│ ├── settings.py
│ ├── urls.py
│ ├── wsgi.py
│ ├── views.py
│ └── celery.py
├── static/
| ├── celery_progress/
| | ├── celery_progress.js - Contains javascript to handle progress bar updates. (External)
| | ├── create_graphs.js - Creates graphs when page loads.
| | └── update_features.js - Updates features on the page once data is loaded.
| ├── css/
| | └── style.css - CSS styling for the webpage.
| ├── img/
| | ├── favicon.ico
| | └── logo_full.png
├── manage.py
└── delete_tables.py - Used to manual delete celery task results from the database. (Django project is configured to delete every 10 minutes)
```

## Future Enhancements
- **Unrestricted Historical News Data**: Plan to integrate more extensive historical news data.
- **Improved Headline Scraping**: Enhancing the efficiency of scraping news headlines.
- **Refining Sentiment Analysis**: Working on improving the accuracy of the sentiment analysis model.
By feeding the sentiment scores to the LSTM may yield better predictive power on closing prices.







   
