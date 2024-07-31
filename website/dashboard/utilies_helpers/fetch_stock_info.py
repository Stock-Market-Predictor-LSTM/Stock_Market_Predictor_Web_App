import yfinance as yf
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from celery.contrib.abortable import AbortableTask
import sys
sys.path.append('../dashboard')
from dashboard.pytorch_models import train_model



@shared_task(bind=True, base = AbortableTask)
def get_close_price(self,data):
    progress_counter = 1
    epochs = 10000

    progress_total = epochs+4
    progress_recorder = ProgressRecorder(self)
    start_date, end_date, ticker= extract_data(data)
    progress_recorder.set_progress(progress_counter, progress_total)
    data_stock = yf.download(ticker, start=start_date, end= end_date)

    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total)
    macd, signal_line = get_macd(data_stock)

    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total)

    x_axis_close_train, y_axis_close_train, x_axis_close_test, y_axis_close_test, c,t = train_model.train(self,data_stock,progress_recorder,progress_counter,progress_total,epochs)
    progress_counter = c
    progress_total = t

    times = [x.strftime('%d-%m-%Y') for x in data_stock.index]
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
                   'ticker':ticker}
    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total)
    
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

