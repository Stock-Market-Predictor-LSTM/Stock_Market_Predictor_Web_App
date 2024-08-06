import yfinance as yf
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from celery.contrib.abortable import AbortableTask
import sys
sys.path.append('../dashboard')
from dashboard.pytorch_models import train_model
import pandas_market_calendars as mcal
import pandas as pd
from dashboard.utilies_helpers.recent_news import get_recent_news


@shared_task(bind=True, base = AbortableTask)
def get_close_price(self,data):
    progress_counter = 1
    epochs = 1000

    progress_total = epochs+4+82
    progress_recorder = ProgressRecorder(self)
    start_date, end_date, ticker= extract_data(data)
    progress_recorder.set_progress(progress_counter, progress_total,description='Downloading Stock Data ...')
    data_stock = yf.download(ticker, start=start_date, end= end_date)

    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total,description='Calculating Indicators ...')
    macd, signal_line = get_macd(data_stock)

    if '26_ema' in data:
        data_stock['26 EMA'] = data_stock['Close'].ewm(span=26, adjust=False).mean()
    if '21_ema' in data:
        data_stock['21 EMA'] = data_stock['Close'].ewm(span=21, adjust=False).mean()
    if '9_ema' in data:
        data_stock['9 EMA'] = data_stock['Close'].ewm(span=9, adjust=False).mean()
    data_stock = data_stock.dropna()
    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total,description='Calculating Indicators ...')

    x_axis_close_train, y_axis_close_train, x_axis_close_test, y_axis_close_test, c,t,features_used,RMSE,corro_features,corrolation_values = train_model.train(self,data_stock,progress_recorder,progress_counter,progress_total,epochs)
    progress_counter = c
    progress_total = t

    news_dates,news_headline,news_sentiments,c, t = get_recent_news(self,progress_recorder,progress_counter,progress_total,ticker)
    progress_counter = c
    progress_total = t
    times = [x.strftime('%d-%m-%Y') for x in data_stock.index]

    next_trading_day = get_next_trading_day(times[-1])
    data_prices = {'dates': times, 
                   'close_prices':list(data_stock['Close']),
                   'volume':list(data_stock['Volume']), 
                   'open_prices':list(data_stock['Open']),
                   'macd':macd,
                   'signal_line':signal_line,
                   'x_axis_close_train':x_axis_close_train,
                   'y_axis_close_train':y_axis_close_train,
                   'x_axis_close_test':x_axis_close_test,
                   'y_axis_close_test':y_axis_close_test,
                   'ticker':ticker,
                   'features_used':features_used,
                   'model_type': 'LSTM',
                   'RMSE':RMSE,
                   'Next_Market_Day': next_trading_day,
                   'news_dates':news_dates,
                   'news_headline':news_headline,
                   'news_sentiments':news_sentiments,
                   'corro_features':corro_features,
                   'corrolation_values':corrolation_values}
    
    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total,description='Loading Data ...')
    
    return data_prices


def extract_data(data):
    start_date = data['start_date']
    end_date = data['end_date']
    ticker = data['ticker']
    return start_date, end_date, ticker

def get_macd(data):
    data_copy = data.copy()
    data_copy['EMA12'] = data_copy['Close'].ewm(span=12, adjust=False).mean()
    data_copy['EMA26'] = data_copy['Close'].ewm(span=26, adjust=False).mean()
    data_copy['MACD'] = data_copy['EMA12'] - data_copy['EMA26']
    data_copy['Signal_Line'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
    return list(data_copy['MACD']), list(data_copy['Signal_Line'])

def get_next_trading_day(previous_trading_day_str):
    # Convert the input to a pandas Timestamp
    previous_trading_day = pd.to_datetime(previous_trading_day_str, format='%d-%m-%Y')
    
    # Get the NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Get the valid trading days around the given previous_trading_day
    schedule = nyse.schedule(start_date=previous_trading_day, end_date=previous_trading_day + pd.Timedelta(days=30))
    
    # Find the next valid trading day
    next_trading_days = schedule[schedule.index > previous_trading_day]
    
    if not next_trading_days.empty:
        return next_trading_days.index[0].date().strftime('%d-%m-%Y')
    else:
        raise ValueError("Unable to find the next trading day within the next 30 days.")

