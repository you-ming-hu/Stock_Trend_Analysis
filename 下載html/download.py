import pathlib
import requests
import time
import pandas as pd
from IPython.display import clear_output

def fetch(drive,category,market,start,end,method,url,param,date_parem_name,date_transfer_func,encoding,sleep,check=False,freq='d'):
    base_path = pathlib.Path(f'{drive}:/').joinpath('股票網頁',category,market)
    base_path.mkdir(parents=True,exist_ok=True)
    dates = pd.date_range(start,end,freq=freq)
    for i,date in enumerate(dates):
        print(i+1,'/',dates.size,'\t',date.date(),end='\t')
        save_path = base_path.joinpath(str(date.date())).with_suffix('.html')
        if save_path.exists():
            if not check:
                print('EXIST')
            else:
                param.update({date_parem_name:date_transfer_func(date)})
                request_method = {'get':requests.get,'post':requests.post}
                resopnse = request_method[method](url,param)
                assert resopnse.status_code == 200
                print('CHECK',end='\t')
                if save_path.read_text(encoding) != resopnse.text.replace('\r','\n'):
                    save_path.rename(save_path.with_suffix('.A.diff.html'))
                    save_path.with_suffix('.B.diff.html').write_text(resopnse.text,encoding=encoding)
                    print('DIFF',end='\t')
                else:
                    print('SAME',end='\t')
                print('SLEEP',end='\t')
                time.sleep(sleep)
                print('SUCCESS!')
        else:
            param.update({date_parem_name:date_transfer_func(date)})
            request_method = {'get':requests.get,'post':requests.post}
            resopnse = request_method[method](url,param)
            assert resopnse.status_code == 200
            save_path.write_text(resopnse.text,encoding=encoding)
            print('SLEEP',end='\t')
            time.sleep(sleep)
            print('SUCCESS!')
        clear_output(wait=True)
    print('FINISH!')
    
def compare(drive,category,market,encoding):
    base_path = pathlib.Path(f'{drive}:/').joinpath('股票網頁',category,market)
    for diff in base_path.glob('*.diff.html'):
        print(diff.name,end='\t')
        print(diff.read_text(encoding) == diff.with_suffix('').with_suffix('').with_suffix('.html').read_text(encoding))