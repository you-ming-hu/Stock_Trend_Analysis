import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from IPython.display import clear_output

last_log_taiex = agent.joinpath('last_update_date_taiex.txt')
last_log_otc = agent.joinpath('last_update_date_otc.txt')
last_log = dict(taiex=last_log_taiex,otc=last_log_otc)

taiex_stage = agent.joinpath('taiex_stage.txt')
otc_stage = agent.joinpath('otc_stage.txt')
stage = dict(taiex=taiex_stage,otc=otc_stage)

taiex_price = agent.joinpath('taiex','price')
taiex_volume = agent.joinpath('taiex','volume')
taiex_html = dict(price=taiex_price,volume=taiex_volume)

otc_price = agent.joinpath('otc','price')
otc_volume = agent.joinpath('otc','volume')
otc_html = dict(price=otc_price,volume=otc_volume)

html_folders = dict(taiex=taiex_html,otc=otc_html)

taiex_merge = agent.joinpath('taiex','merge')
otc_merge = agent.joinpath('otc','merge')
merge_folders = dict(taiex=taiex_merge,otc=otc_merge)

taiex_price.mkdir(parents=True,exist_ok=True)
taiex_volume.mkdir(parents=True,exist_ok=True)
taiex_merge.mkdir(parents=True,exist_ok=True)
otc_price.mkdir(parents=True,exist_ok=True)
otc_volume.mkdir(parents=True,exist_ok=True)
otc_merge.mkdir(parents=True,exist_ok=True)


def fetch(cate,which,date):
    date = pd.Timestamp(date)
    price_url = dict(
        taiex='https://www.twse.com.tw/indicesReport/MI_5MINS_HIST',
        otc='https://www.tpex.org.tw/web/stock/index_info/Inxh/Inx_result.php')
    volume_url = dict(
        taiex='https://www.twse.com.tw/exchangeReport/FMTQIK',
        otc='https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_index/st41_result.php')
    url = dict(price=price_url,volume=volume_url)
    param = dict(
        taiex=dict(response='html',date=date.strftime(f'%Y%m%d')),
        otc=dict(l='zh-tw',s='0,asc,0',o='htm',d=date.strftime(f'{date.year-1911}/%m')))
    res = requests.get(url[cate][which], param[which])
    assert res.status_code == 200
    return res.text

def get_html(cate,which,date):
    date = str(pd.Timestamp(date).date())
    print(f'{date} GET {which:8} {cate} html:', end='\t')
    path = html_folders[which][cate].joinpath(date).with_suffix('.html')
    if path.exists():
        print('EXIST', end='\t')
    else:
        print('REQUEST', end='\t')
        text = fetch(cate,which,date)
        path.write_text(text,encoding='utf8')   
    print('SUCCESS!')
    return None

def get_all_html(which,dates):
    for date in dates:
        get_html('price',which,date)
        time.sleep(10)
        get_html('volume',which,date)
        time.sleep(10)
    return None

def verify_html(cate,which,date):
    date = str(pd.Timestamp(date).date())
    print(f'VERIFY {date} {which:8} {cate} html:', end='\t')
    path = html_folders[which][cate].joinpath(date).with_suffix('.html')
    text_from_path = path.read_text(encoding='utf8')
    text_from_url = fetch(cate,which,date)
    if text_from_path != text_from_url.replace('\r','\n'):
        print('DIFFERENT',end='\t')
        time.sleep(10)
        verify_text = fetch(cate,which,date)
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
                verify_text_2 = fetch(cate,which,date)
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

def verify_all_html(which,dates):
    for date in dates:
        verify_html('price',which,date)
        time.sleep(10)
        verify_html('volume',which,date)
        time.sleep(10)
    return None

def parse_taiex_html(date):
    date = pd.Timestamp(date)

    price_path = taiex_price.joinpath(str(date.date())).with_suffix('.html')
    price = pd.read_html(price_path)[0]
    assert date.strftime(f'{date.year-1911}年%m月 發行量加權股價指數歷史資料') == price.columns.levels[0]
    price.columns = price.columns.droplevel(0)

    volume_path = taiex_volume.joinpath(str(date.date())).with_suffix('.html')
    volume = pd.read_html(volume_path)[0]
    assert date.strftime(f'{date.year-1911}年%m月市場成交資訊') == volume.columns.levels[0]
    volume.columns = volume.columns.droplevel(0)

    return price,volume

def parse_otc_html(date):
    date = pd.Timestamp(date)

    price_path = otc_price.joinpath(str(date.date())).with_suffix('.html')
    price_bs = BeautifulSoup(price_path.read_text(encoding='utf8'))
    price_bs.find('tfoot').decompose()
    price = pd.read_html(price_bs.prettify())[0]
    assert date.strftime(f'櫃買指數(月查詢) 資料年月:{date.year-1911}/%m') == price.columns.levels[0][0].replace('\xa0',' ')
    price.columns = price.columns.droplevel(0)

    volume_path = otc_volume.joinpath(str(date.date())).with_suffix('.html')
    volume_bs = BeautifulSoup(volume_path.read_text(encoding='utf8'))
    volume_bs.find('tfoot').decompose()
    volume = pd.read_html(volume_bs.prettify())[0]
    assert date.strftime(f'日成交量值指數 資料年月:{date.year-1911}/%m') == volume.columns.levels[0][0].replace('\xa0',' ')
    volume.columns = volume.columns.droplevel(0)
    
    return price,volume

def read_taiex_html(date):
    date = str(pd.Timestamp(date).date())
    price, volume = parse_taiex_html(date)

    price[['year','month','day']] = price['日期'].str.split('/',expand=True)
    price.year = (price.year.astype(int)+1911).astype(str)
    price['date'] = pd.to_datetime(price[['year','month','day']])
    price = price.rename({'開盤指數':'open','最高指數':'high','最低指數':'low','收盤指數':'close'},axis=1,errors='raise')
    price = price[['date','open','high','low','close']]

    volume[['year','month','day']] = volume['日期'].str.split('/',expand=True)
    volume.year = (volume.year.astype(int)+1911).astype(str)
    volume['date'] = pd.to_datetime(volume[['year','month','day']])
    volume = volume.rename({'成交股數':'volume','成交金額':'capital','成交筆數':'count','發行量加權股價指數':'close'},axis=1,errors='raise')
    volume = volume[['date','close','volume','count','capital']]
    return price, volume

def read_otc_html(date):
    date = str(pd.Timestamp(date).date())
    price, volume = parse_otc_html(date)

    price['date'] = pd.to_datetime(price['日期'])
    price = price.rename({'開市':'open','最高':'high','最低':'low','收市':'close'},axis=1,errors='raise')
    price = price[['date','open','high','low','close']]

    volume[['year','month','day']] = volume['日 期'].str.split('/',expand=True)
    volume.year = (volume.year.astype(int)+1911).astype(str)
    volume['date'] = pd.to_datetime(volume[['year','month','day']])
    volume = volume.rename({'成交股數（仟股）':'volume','金額（仟元）':'capital','筆數':'count','櫃買指數':'close'},axis=1,errors='raise')
    volume[['volume','capital']] = volume[['volume','capital']]*1000
    volume = volume[['date','close','volume','count','capital']]
    return price, volume

def merge_html_to_csv(which,date):
    print(f'PARSE {which:8} html',end='\t')
    date = str(pd.Timestamp(date).date())
    read_html = dict(taiex=read_taiex_html,otc=read_otc_html)
    path = merge_folders[which].joinpath(date).with_suffix('.csv')
    price,volume = read_html[which](date)
    price[['volume','count','capital']] = volume[['volume','count','capital']]
    price = price.set_index('date')
    price.to_csv(path.as_posix(),encoding='utf8')
    return price

def read_csv(which,date):
    date = str(pd.Timestamp(date).date())
    path = merge_folders[which].joinpath(date).with_suffix('.csv')
    if path.exists():
        print(f'{which:8} CSV EXIST!',end='\t')
        table = pd.read_csv(path.as_posix(),encoding='utf8',index_col='date',parse_dates=True)
    else:
        table = merge_html_to_csv(which,date)
    return table

def query(which,dates):
    new_data = pd.DataFrame()
    for date in dates:
        print(f'QUERY {date.date()} :', end='\t')
        new_data = new_data.append(read_csv(which,date))
        print('SUCCESS!')
    return new_data

def acquire(which,dates,last_update_date):
    print(f'ACQUIRE {which}', end='\t')
    new_data = query(which,dates)
    new_data = new_data.loc[last_update_date + pd.Timedelta(days=1):]
    print('SUCCESS!')
    return new_data

def append(which,new_data):
    if not new_data.empty:
        path = depository.joinpath(which,category,'data.csv')
        print('APPEND')
        print(path.as_posix())
        new_data.to_csv(path.as_posix(),header=False,encoding='utf8',mode='a')
        last_log[which].write_text(str(new_data.index[-1].date()))
    else:
        print('no new data')
    clear_output(wait=True)

def update_general(which,end_date='today'):
    last_update_date = pd.Timestamp(last_log[which].read_text())
    scheduled_dates = pd.date_range(last_update_date.replace(day=1), end_date, freq='MS')

    if stage[which].read_text() == 'html':
        print('='*50,'FETCH HTML','='*50)
        get_all_html(which,scheduled_dates)
        stage[which].write_text('verify')
        clear_output(wait=True)

    if stage[which].read_text() == 'verify':
        print('='*50,'VERIFY HTML','='*50)
        verify_all_html(which,scheduled_dates)
        stage[which].write_text('append')
        clear_output(wait=True)

    if stage[which].read_text() == 'append':
        print('='*50,'APPEND','='*50)
        new_data = acquire(which,scheduled_dates,last_update_date)
        append(which,new_data)
        stage[which].write_text('html')
        clear_output(wait=True)

def release():
    release_folders = [taiex_price, taiex_volume, otc_price, otc_volume, taiex_merge, otc_merge]
    for folder in release_folders:
        for item in folder.iterdir():
            print(item.as_posix())
            item.unlink()
            clear_output(wait=True)

def update(end_date='today'):
    update_general('taiex',end_date)
    update_general('otc',end_date)
    release()
    print('FINISH!')
