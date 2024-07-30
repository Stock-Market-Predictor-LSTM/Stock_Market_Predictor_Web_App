import yfinance as yf
from celery import shared_task
from celery_progress.backend import ProgressRecorder


@shared_task(bind=True)
def get_close_price(self,data):
    progress_recorder = ProgressRecorder(self)
    print('0')
    start_date, end_date, ticker= extract_data(data)

    progress_recorder.set_progress(1, 4)
    print('1')
    data_stock = yf.download(ticker, start=start_date, end= end_date)

    progress_recorder.set_progress(2, 4)
    print('2')
    macd, signal_line = get_macd(data_stock)
    
    progress_recorder.set_progress(3, 4)

    times = [x.strftime('%d-%m-%Y') for x in data_stock.index]
    data_prices = {'dates': times, 
                   'close_prices':list(data_stock['Close']),
                   'volume':list(data_stock['Volume']), 
                   'open_prices':list(data_stock['Open']),
                   'macd':macd,
                   'signal_line':signal_line,
                   'ticker':ticker}
    
    progress_recorder.set_progress(4, 4)
    print('done')
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

