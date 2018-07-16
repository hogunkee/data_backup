import os
import numpy as np
import datetime
import pandas
from scipy import stats
from keras.preprocessing.text import Tokenizer
from pymongo import MongoClient


GLOVE_DIR = '../data/glove.6B/'
#EMBEDDING_DIR = '../../data/wiki.ko.vec'

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']
client = MongoClient('localhost', 27017)
db = client['hackathon']
d = '2018-07-09T:00:00:00Z'

## load news data ##
def load_text(IDX):
    company = STOCK_LIST[IDX].lower()
    rows = db.news.find({'symbol': company, 'date': {'$lt':d}}, {'date': 1, 'title': 1, 'text': 1}).sort('date')
    #rows = db.news.find({'symbol': company}, {'date': 1, 'title': 1, 'text': 1}).sort('date')

    company_data = []
    for row in list(rows):
        _data = [row['date'].split('T')[0], row['title'], row['text']]
        #_data = list(row.values())[1:]
        company_data.append(_data)

    dates, texts, titles = zip(*company_data)
    return dates, texts, titles

## load stock data ##
def split_date_stock(line):
    date = line.split(',')[0]
    #date = list(map(lambda k: int(k), line.split(',')[0].split('-')))
    stock = float(line.split(',')[4].replace('\n',''))
    return date, stock

def load_stock(stock_dir):
    f = open(stock_dir, 'r')
    lines = f.readlines()[1:]
    lines = sorted(list(map(lambda k: split_date_stock(k), lines)), key=lambda q: q[0])
    date, stock = zip(*lines)
    return date, stock

def get_stock_slope(date, stock):
    # input date form: '2015-01-01'
    assert len(date)==len(stock)
    tslope = [0 for _ in range(len(date)-4)]
    split_date = list(map(lambda x: datetime.date(x[0],x[1],x[2]), \
            list(map(lambda d: list(map(lambda k: int(k), d.split('-'))), date))))
    for i in range(len(date)-4):
        x = list(map(lambda d: (d-split_date[i]).days, split_date[i:i+5]))
        y = list(map(lambda s: 100*s/stock[i], stock[i: i+5]))
        tslope[i] = stats.linregress(x,y)[0]

    c = 0
    slope = []
    updown = []
    daterange = pandas.date_range(date[0], date[-1])
    wholedate = list(map(lambda d: str(d).split(' ')[0], daterange))
    for _date in wholedate:
        if (date[c] < _date):
            c+=1
            if c>=len(tslope):
                break
        slope.append(tslope[c])
        updown.append(1 if tslope[c]>0 else 0)

    return list(zip(slope, wholedate[:len(slope)+1], updown))

def stock_data(stock_idx):
    result = []
    company = STOCK_LIST[stock_idx]
    stock_dir = '../data/' + company + '.csv'
    date, stock = load_stock(stock_dir)
    return get_stock_slope(date, stock)

## embedding mat ##
def make_embedding(tokenizer, voca_size, em_dim):
    pre_embedding = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(em_dim)+'d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        pre_embedding[word] = coefs
    f.close()

    word2index = tokenizer.word_index
    n_symbols = min(voca_size, len(word2index))

    ## new embedding
    x,y=0,0
    embedding_mat = np.zeros((n_symbols, em_dim))
    for _word, _idx in word2index.items():
        if _idx < n_symbols:
            if _word in pre_embedding.keys():
                embedding_mat[_idx] = pre_embedding[_word]
                x+=1
            else:
                embedding_mat[_idx] = np.random.normal(0, 0.1, em_dim)
                y+=1
    print('embedding:', x)
    print('no matching:', y)

    return embedding_mat
