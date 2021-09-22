import requests
from bs4 import BeautifulSoup
import pandas as pd
import pathlib
import datetime

def fetch(which):
    url = 'https://isin.twse.com.tw/isin/C_public.jsp'
    param = dict(
        exchange={'strMode':'2'},
        otc={'strMode':'4'})
    response = requests.get(url, param[which])
    assert response.status_code == 200
    return response.text

def get_html(which):
    print(f'overview GET {which:8} html:', end='\t')
    path = agent.joinpath(which).with_suffix('.html')
    if path.exists():
        print('EXIST', end='\t')
    else:
        print('REQUEST', end='\t')
        text = fetch(which)
        path.write_text(text,encoding='utf8')   
    print('SUCCESS!')
    return None

def read_html(which):
    path = agent.joinpath(which).with_suffix('.html')
    text = path.read_text(encoding='utf8')
    soup = BeautifulSoup(text, 'lxml')

    labels = [''.join(t.text.split())
                for t in soup.find_all('b', text=lambda x: x != ' ')]

    table = pd.read_html(text, header=0,converters={})[0]
    table.loc[table['備註'].isin(labels), 'labels'] = labels
    table['labels'] = table['labels'].ffill()
    table = table.loc[~table['備註'].isin(labels)]

    split = table['有價證券代號及名稱'].str.split(expand=True)

    table['ID'] = split[0]
    
    table['name'] = split.loc[:,1:].fillna('').apply(lambda row: ''.join(row),axis=1)

    table['is_stock'] = table['labels'] == '股票'
    
    table['industry'] = table['產業別']
    table['market'] = which
    
    table = table[['ID', 'name', 'is_stock','industry','market']]
    
    table = table.reset_index(drop=True)
    return table

def html_to_csv(which):
    print(f'PARSE {which:8} html',end='\t')
    path = agent.joinpath(which).with_suffix('.csv')
    table = read_html(which)
    table.to_csv(path.as_posix(),encoding='utf8')
    print('SUCCESS!')
    return table

def read_csv(which):
    path = agent.joinpath(which).with_suffix('.csv')
    if path.exists():
        print(f'{which:8} CSV EXIST!')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = html_to_csv(which)
    return table

def merge_csv():
    path = agent.joinpath('merge').with_suffix('.csv')
    exchange = read_csv('exchange')
    otc = read_csv('otc')
    print('MERGE')
    table = exchange.append(otc, ignore_index=False)
    duplicate = table.loc[table.ID.duplicated(False)]
    if not duplicate.empty:
        print('\n','='*25,'合併上市上櫃一覽表時發現重複代碼','='*25)
        display(duplicate)
        assert False, f'duplicate: {duplicate.ID.unique()}'
    table = table.reset_index(drop=True)
    table.to_csv(path.as_posix(),encoding='utf8')
    return table

def read_merge():
    path = agent.joinpath('merge').with_suffix('.csv')
    if path.exists():
        print('MERGE CSV EXIST!')
        table = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str},index_col=0)
    else:
        table = merge_csv()
    return table

def release():
    release_file = ['*.csv','*.html']
    for file in release_file:
        for item in agent.glob(file):
            item.unlink()

def update():
    print('='*10,str(datetime.date.today()),'='*10)
    get_html('exchange')
    get_html('otc')
    path = depository.joinpath('data.csv')
    new_data = read_merge()
    new_data['status'] = 'new'
    print('UPDATE')
    if path.exists():
        previous_data = pd.read_csv(path.as_posix(),encoding='utf8',dtype={'ID':str}, index_col=0)
        previous_data['status'] = 'data'
        
        merge_data = previous_data.append(new_data, ignore_index=False)
        merge_data = merge_data.drop_duplicates(subset=['ID','name','is_stock','market'])
        
        duplicate = merge_data.loc[merge_data.ID.duplicated(False)]
        if not duplicate.empty:
            print('\n','='*25,'一覽表添加新資料時發現內容不重複，但代碼重複的資料','='*25)
            display(duplicate)
            assert False, f'duplicate: {duplicate.ID.unique()}'
    else:
        merge_data = new_data
    merge_data = merge_data.reset_index(drop=True)
    merge_data = merge_data.drop('status',axis=1)
    merge_data.to_csv(path,encoding='utf8')
    print('RELEASE')
    release()
    print('FINISH!')