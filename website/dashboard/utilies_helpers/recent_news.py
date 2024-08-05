import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from pathlib import Path
import os
# Build paths inside the project like this: BASE_DIR / 'subdir'.
folder = Path(__file__).resolve().parent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_sentiment(model, tokenizer, headline, entity, max_len):
  sentiment_to_label = {0: 1, 1: 0, 2: -1}
  model.eval()
  inputs = tokenizer.encode_plus(
      entity,
      headline,
      add_special_tokens=True,
      max_length=max_len,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
  )

  input_ids = inputs['input_ids'].to(device)
  attention_mask = inputs['attention_mask'].to(device)

  with torch.no_grad():
      output = model(input_ids, attention_mask)
      _, predicted = torch.max(output, 1)

  return sentiment_to_label[predicted.cpu().numpy()[0]]

class EntitySentimentModel(nn.Module):
    def __init__(self, n_classes):
        super(EntitySentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)




def get_recent_news(async_data,progress_recorder,progress_counter,progress_total,ticker):
    tokenizer = BertTokenizer.from_pretrained(f'{folder}/tokenizer_directoryv4')
    model = EntitySentimentModel(n_classes=3)
    state_dict = torch.load(f'{folder}/entire_modelv4_state_dict.pth', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    options = Options()
    options.add_argument("--headless")
    #options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total,description='Loading News Data ...')
    #driver = webdriver.Firefox(service = FirefoxService(f'{folder}/geckodriver.exe'),options = options)
    driver = webdriver.Firefox(options = options)
    wait = WebDriverWait(driver, 10)
    progress_counter+= 1
    progress_recorder.set_progress(progress_counter, progress_total,description='Loading News Data ...')
    driver.get(f'https://www.benzinga.com/quote/{ticker.upper()}/news')

    div_elements = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "content-title")))
    dates = []
    headline = []
    for i, div in enumerate(div_elements):
        if async_data.is_aborted():
            break
        progress_counter+= 1
        if i % 10 == 0:
            progress_recorder.set_progress(progress_counter, progress_total,description='Loading News Data ...')
        # Use a try-except block to handle any potential issues with finding elements
        try:
            span = div.find_element(By.TAG_NAME, 'span')
            parent_div = div.find_element(By.XPATH, '..')
            outer_html = parent_div.get_attribute('outerHTML')
            
            # Check if parent div has the sponsored tag
            if 'news-item-sponsored-tag' not in outer_html:
                # Check if the span does not contain '...'
                if '...' not in span.get_attribute('innerHTML'):
                    if ticker.lower() == 'aapl':
                        if 'apple' in span.get_attribute('innerHTML').lower():
                            # Extract date and headline text efficiently
                            date_element = parent_div.find_element(By.CSS_SELECTOR, '.text-gray-500.content-headline-datetime')
                            
                            # Store parsed date and headline
                            dates.append(date_element.get_attribute('innerHTML'))
                            headline.append(span.get_attribute('innerHTML'))
                    elif ticker.lower() == 'nvda':
                         if 'nvidia' in span.get_attribute('innerHTML').lower():
                            # Extract date and headline text efficiently
                            date_element = parent_div.find_element(By.CSS_SELECTOR, '.text-gray-500.content-headline-datetime')
                            
                            # Store parsed date and headline
                            dates.append(date_element.get_attribute('innerHTML'))
                            headline.append(span.get_attribute('innerHTML'))
                    elif ticker.lower() == 'goog':
                        if 'google' in span.get_attribute('innerHTML').lower() or 'alphabet' in span.get_attribute('innerHTML').lower():
                            # Extract date and headline text efficiently
                            date_element = parent_div.find_element(By.CSS_SELECTOR, '.text-gray-500.content-headline-datetime')
                            
                            # Store parsed date and headline
                            dates.append(date_element.get_attribute('innerHTML'))
                            headline.append(span.get_attribute('innerHTML'))
                        

        except Exception as e:
            print(f"Error processing element {i}: {e}")
    driver.quit()
    progress_recorder.set_progress(progress_counter, progress_total,description='Loading News Data ...')
    progress_total = progress_counter + 1 + 8
    progress_recorder.set_progress(progress_counter, progress_total,description='Evaluating Sentiment ...')
    while len(headline) < 8:
        if async_data.is_aborted():
            break
        headline.append('')
    sentiments = []
    entity = ''
    for i in range(8):
        if async_data.is_aborted():
            break
        progress_counter+= 1
        progress_recorder.set_progress(progress_counter, progress_total,description='Evaluating Sentiment ...')
        entity = ''
        if headline[i] != '':
            if ticker.lower() == 'aapl':
                entity = 'apple'
                sentiment = predict_sentiment(model, tokenizer, headline[i], entity, 512)
                sentiments.append(sentiment)
            elif ticker.lower() == 'nvda':
                entity = 'nvidia'
                sentiment = predict_sentiment(model, tokenizer, headline[i], entity, 512)
                sentiments.append(sentiment)
            elif ticker.lower() == 'goog':
                if 'google' in headline[i].lower():
                    entity = 'google'
                    sentiment = predict_sentiment(model, tokenizer, headline[i], entity, 512)
                    sentiments.append(sentiment)
                elif 'alphabet' in headline[i].lower():
                    entity = 'alphabet'
                    sentiment = predict_sentiment(model, tokenizer, headline[i], entity, 512)
                    sentiments.append(sentiment)
            
        else:
            sentiments.append(0)
    progress_total = progress_counter + 1 
    progress_recorder.set_progress(progress_counter, progress_total,description='Loading Data ...')
    return dates[:8],headline[:8],sentiments, progress_counter, progress_total