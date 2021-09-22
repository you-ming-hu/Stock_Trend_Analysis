import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from IPython.display import clear_output

last_log = agent.joinpath('last_update_date.txt')
active_stocks = agent.joinpath('active_stocks.csv')
stage = agent.joinpath('stage.txt')

exchange_folder = agent.joinpath('exchange')
otc_folder = agent.joinpath('otc')
merge_folder = agent.joinpath('merge')
authorized_folder = agent.joinpath('authorized')
unauthorized_folder = agent.joinpath('unauthorized')
error_folder = agent.joinpath('error')
new_folder = agent.joinpath('new')

exchange_folder.mkdir(exist_ok=True)
otc_folder.mkdir(exist_ok=True)
merge_folder.mkdir(exist_ok=True)
authorized_folder.mkdir(exist_ok=True)
unauthorized_folder.mkdir(exist_ok=True)
error_folder.mkdir(exist_ok=True)
new_folder.mkdir(exist_ok=True)

market_folders = dict(exchange=exchange_folder,otc=otc_folder)

output_columns = ['date','open','high','low','close','volume','count','capital','ID','name','market']

def fetch(which,date):
    date = pd.Timestamp(date)
    url=dict(
        exchange='https://www.twse.com.tw/exchangeReport/MI_INDEX',
        otc='https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php')
    param = dict(
        exchange={'response': 'html', 'type': 'ALLBUT0999', 'date': date.strftime('%Y%m%d')},
        otc={'l': 'zh-tw', 'o': 'htm', 'se': 'EW', 's': '0,asc,0', 'd': date.strftime(f'{date.year-1911}/%m/%d')})
    response = requests.get(url[which], param[which])
    assert response.status_code == 200
    return response.text

def get_html(which,date):
    date = str(pd.Timestamp(date).date())
    print(f'{date} GET {which:8} html:', end='\t')
    path = market_folders[which].joinpath(date).with_suffix('.html')
    if path.exists():
        print('EXIST', end='\t')
    else:
        print('REQUEST', end='\t')
        text = fetch(which,date)
        path.write_text(text,encoding='utf8')   
    print('SUCCESS!')
    return None

def get_all_html(dates):
    for date in dates:
        get_html('exchange',date)
        get_html('otc',date)
        time.sleep(10)
    return None

def verify_html(which,date):
    date = str(pd.Timestamp(date).date())
    print(f'VERIFY {date} {which:8} html:', end='\t')
    path = market_folders[which].joinpath(date).with_suffix('.html')
    text_from_path = path.read_text(encoding='utf8')
    text_from_url = fetch(which,date)
    if text_from_path != text_from_url.replace('\r','\n'):
        print('DIFFERENT',end='\t')
        time.sleep(10)
        verify_text = fetch(which,date)
        if verify_text == text_from_path:
            print('KEEP FIRST',end='\t')
        elif verify_text == text_from_url:
            print('KEEP SECOND',end='\t')
            path.write_text(text_from_url,encoding='utf8')
        else:
            attempt = 1
            print('ATTEMPT: ',end=' ')
            while True:
                print(attempt,end=' ')
                time.sleep(10)
                verify_text_2 = fetch(which,date)
                if verify_text == verify_text_2:
                    path.write_text(verify_text_2,encoding='utf8')
                    break
                else:
                    verify_text = verify_text_2
                    attempt+=1
    else:
        print('SAME',end='\t')
    print('SUCCESS')
    return None

def verify_all_html(dates):
    for date in dates:
        verify_html('exchange',date)
        verify_html('otc',date)
        time.sleep(10)
    return None

def parse_exchange_html(date):
    date = pd.Timestamp(date)
    text = exchange_folder.joinpath(str(date.date())).with_suffix('.html').read_text(encoding='utf8')
    if '沒有符合條件的資料' in text:
        table = pd.DataFrame()
    else:
        bs = BeautifulSoup(text, 'lxml')
        title = f'{date.year-1911}年{date.month:0>2}月{date.day:0>2}日每日收盤行情(全部(不含權證、牛熊證))'
        bs = bs.find(text=title).parent.parent.parent.parent
        table = pd.read_html(bs.prettify(),converters={'證券代號':str})[0]
        table.columns = table.columns.droplevel([0, 1])
    return table

def parse_otc_html(date):
    date = pd.Timestamp(date)
    text = otc_folder.joinpath(str(date.date())).with_suffix('.html').read_text(encoding='utf8')
    if '共0筆' in text:
        table = pd.DataFrame()
    else:
        assert date.strftime(f'{date.year-1911}/%m/%d') in text
        bs = BeautifulSoup(text, 'lxml')
        bs.find('tr').decompose()
        bs.find('tfoot').decompose()
        table = pd.read_html(bs.prettify(),converters={'代號':str})[0]
    return table

def read_html(which,date):
    date = str(pd.Timestamp(date).date())
    parse_html_method = dict(exchange=parse_exchange_html,otc=parse_otc_html)
    translation = dict(
        exchange={
            '證券代號': 'ID', '證券名稱': 'name',
            '成交股數': 'volume', '成交筆數': 'count', '成交金額': 'capital',
            '開盤價': 'open', '最高價': 'high', '最低價': 'low', '收盤價': 'close'
            },
        otc={
            '代號': 'ID', '名稱': 'name',
            '成交股數': 'volume', '成交金額(元)': 'capital', '成交筆數': 'count',
            '收盤': 'close', '開盤': 'open', '最高': 'high','最低': 'low'
            }
        )
    table = parse_html_method[which](date)
    if table.empty:
        table = pd.DataFrame(columns=output_columns)
    else:
        table = table.rename(translation[which], axis=1)
        table['date'] = date
        table['market'] = which
        table = table[output_columns]
    return table

def html_to_csv(which,date):
    print(f'PARSE {which:8} html',end='\t')
    date = str(pd.Timestamp(date).date())
    path = market_folders[which].joinpath(date).with_suffix('.csv')
    table = read_html(which,date)
    if not table.empty:
        table.to_csv(path.as_posix(),encoding='utf8')
    return table

def read_csv(which,date):
    date = str(pd.Timestamp(date).date())
    path = market_folders[which].joinpath(date).with_suffix('.csv')
    if path.exists():
        print(f'{which:8} CSV EXIST!',end='\t')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = html_to_csv(which,date)
    return table

def merge_csv(date):
    date = str(pd.Timestamp(date).date())
    path = merge_folder.joinpath(date).with_suffix('.csv')
    exchange = read_csv('exchange',date)
    otc = read_csv('otc',date)
    print('MERGE',end='\t')
    if not exchange.empty and not otc.empty:
        table = exchange.append(otc)
        table.to_csv(path.as_posix(),encoding='utf8')
    else:
        table = pd.DataFrame()
    return table

def read_merge(date):
    date = str(pd.Timestamp(date).date())
    path = merge_folder.joinpath(date).with_suffix('.csv')
    if path.exists():
        print('MERGE CSV EXIST!',end='\t')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = merge_csv(date)
    return table

def abnormal_classfier(row):
    try:
        format_row = row[['open','high','low','close','volume','count','capital']].astype(float)
        ohlc = format_row[['open','high','low','close']]
        lowest = ohlc.min()
        highest =  ohlc.max()
        if ohlc['low'] != lowest:
            return 'abnormal_low'
        elif ohlc['high'] != highest:
            return 'abnormal_high'
        elif format_row.isnull().any():
            return 'abnormal_null'
        else:
            return ''
    except ValueError:
        return 'abnormal_format'

def classify(date):
    date = str(pd.Timestamp(date).date())
    authorize_path = authorized_folder.joinpath(date).with_suffix('.csv')
    unauthorize_path = unauthorized_folder.joinpath(date).with_suffix('.csv')
    table = read_merge(date)
    print('CLASSIFY',end='\t')
    if not table.empty:
        table = table.set_index('ID')
        table['unauthorized'] = ''
        
        table.loc[table.index.duplicated(False),'unauthorized'] += 'duplicate, '
        table.loc[table.index.difference(overview.ID),'unauthorized'] += 'exceed, '
        table.loc[:,'unauthorized'] += table.apply(abnormal_classfier,axis=1)
        
        authorized = table.loc[table['unauthorized']==''].drop('unauthorized',axis=1).reset_index()
        unauthorized = table.loc[table['unauthorized']!=''].sort_index().reset_index()
        authorized.to_csv(authorize_path.as_posix(),encoding='utf8')
        unauthorized.to_csv(unauthorize_path.as_posix(),encoding='utf8')
    else:
        authorized = pd.DataFrame()
        unauthorized = pd.DataFrame()
    return authorized,unauthorized

def read_authorized(date):
    date = str(pd.Timestamp(date).date())
    path = authorized_folder.joinpath(date).with_suffix('.csv')
    if path.exists():
        print('AUTHORIZED CSV EXIST!',end='\t')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = classify(date)[0]
    return table

def read_unauthorized(date):
    date = str(pd.Timestamp(date).date())
    path = unauthorized_folder.joinpath(date).with_suffix('.csv')
    if path.exists():
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = classify(date)[1]
    return table

def find_error(date):
    date = str(pd.Timestamp(date).date())
    path = unauthorized_folder.joinpath(date).with_suffix('.csv')
    table = read_unauthorized(date)
    if not table.empty:
        table = table.loc[(table[['open','high','low','close']] != '--').all(axis=1)]
        table = table.loc[(table[['open','high','low','close']] != '----').all(axis=1)]
        if not table.empty:
            table.to_csv(path.as_posix(),encoding='utf8')
    return table

def gether_error():
    error = pd.DataFrame()
    for path in  sorted(error_folder.glob('*.csv')):
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
        error = error.appen(table)
    return error

def upload(date):
    date = str(pd.Timestamp(date).date())
    path = new_folder.joinpath(date).with_suffix('.csv')
    stock_id = overview.loc[overview.is_stock,'ID']
    table = read_authorized(date)
    print('UPLOAD',end='\t')
    if not table.empty: 
        table = table.set_index('ID')
        table = table.loc[table.index.intersection(stock_id)]
        table = table.reset_index()
        table.to_csv(path.as_posix(),encoding='utf8')
    return table
    
def query(date):
    date = str(pd.Timestamp(date).date())
    path = new_folder.joinpath(date).with_suffix('.csv')
    if path.exists():
        print('NEW CSV EXIST!',end='\t')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = upload(date)
    return table

def acquire(dates):
    new_data = pd.DataFrame()
    for date in dates:
        date = str(pd.Timestamp(date).date())
        print(f'ACQUIRE {date} :', end='\t')
        table = query(date)
        exchange_csv_exist = exchange_folder.joinpath(date).with_suffix('.csv').exists()
        otc_csv_exist = otc_folder.joinpath(date).with_suffix('.csv').exists()
        if (not exchange_csv_exist and otc_csv_exist) or (exchange_csv_exist and not otc_csv_exist):
            #阻止跳躍性更新的問題，只要更新過程有發生上市上櫃資料不同步就會終止更新
            print('UNPAIRED CSV ERROR!')
            break
        else:
            if table.empty:
                print('EMPTY!',end='\t')
            else:
                new_data = new_data.append(table)
                table['ID'].to_csv(active_stocks.as_posix(),encoding='utf8')
                last_date = date
            print('SUCCESS!')
    return last_date, new_data
            
def archive(table):
    table = table.set_index('ID')
    stocks = sorted(table.index.unique())
    for stock in stocks:
        path = depository.joinpath(stock,category)
        print('ARCHIVE')
        print(path.as_posix())
        path.mkdir(parents=True,exist_ok=True)
        data = table.loc[[stock]]
        data = data.reset_index()
        data = data.sort_values('date')
        data = data[output_columns]
        data.to_csv(path.joinpath('new.csv').as_posix(),index=False,encoding='utf8')
        clear_output(wait=True)

def append():
    for new in sorted(depository.glob('/'.join(['*',category.joinpath('new.csv').as_posix()]))):
        print('APPEND')
        print(new.as_posix())
        data = new.with_name('data.csv')
        if data.exists():
            table = pd.read_csv(new.as_posix(),encoding='utf8',dtype={'ID':str})
            table.to_csv(data.as_posix(),header=False,index=False,encoding='utf8',mode='a')
            new.unlink()
        else:
            new.rename(data)
        clear_output(wait=True)

def release():
    release_folders = [exchange_folder,otc_folder,merge_folder,authorized_folder,new_folder]
    for folder in release_folders:
        for item in folder.iterdir():
            print(item.as_posix())
            item.unlink()
            clear_output(wait=True)

def update(end_date='today'):
    last_update_date = last_log.read_text()
    scheduled_dates = pd.date_range(last_update_date, end_date)
    if scheduled_dates.size == 1:
        print(f'已經下載到:{scheduled_dates[0].date()},沒有需要下載的日期')
    else:
        scheduled_dates = scheduled_dates[1:]
        
        if stage.read_text() == 'html':
            print('='*50,'FETCH HTML','='*50)
            get_all_html(scheduled_dates)
            stage.write_text('verify')
            clear_output(wait=True)
            
        if stage.read_text() == 'verify':
            print('='*50,'VERIFY HTML','='*50)
            verify_all_html(scheduled_dates)
            stage.write_text('archive')
            clear_output(wait=True)
            
        if stage.read_text() == 'archive':
            print('='*50,'ACQUIRE','='*50)
            last_update_date, new_data = acquire(scheduled_dates)
            clear_output(wait=True)
            print('='*50,'ARCHIVE','='*50)
            archive(new_data)
            last_log.write_text(last_update_date)
            stage.write_text('append')
            clear_output(wait=True)
            
        if stage.read_text() == 'append':
            print('='*50,'APPEND','='*50)
            append()
            stage.write_text('release')
            clear_output(wait=True)
            
        if stage.read_text() == 'release':
            print('='*50,'RELEASE','='*50)
            release()
            stage.write_text('html')
            clear_output(wait=True)
            
        error = gether_error()
        if not error.empty:
            print('ERROR TABLE')
            display(error)
        
        print('FINISH!')

    