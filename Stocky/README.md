# Stocky

主程式碼與資料存放處

## 結構

```
├─ Database					資料存放處，agent負責管理depository內的資料
|  ├─ agent
|  |  └─ ...
|  └─ depository
|     └─ ...
├─ Analyser.py				分析所需的工具
├─ Database.py				維護Database的API
├─ Dataset.py				讀取Database的API
├─ Filter.py				處理布林訊號，過濾或推移訊號等功能
├─ FloatingIndicator.py		股票常用指標如MA, MACD, RSI等，輸出較低資料頻率每日的運算結果
├─ Indicator.py				股票常用指標，僅輸出較低資料最頻率最末日運算結果。
├─ Resample.py				重新採樣資料頻率
├─ Signalizer.py			將數值類的指標等數值轉化為布林訊號
└─ Structure.py				建立模型所需的框架Model, Estimator, Detector, Trader, Adjuster
```

## Database

依股票類型可分為: 1.總體(general) 2.個股(individual)

---

依分析面向可分為: 1.技術面(technical) 2.籌碼面(chip) 3.基本面(fundmental) ，三個面相又可再細分為以下子項目:

1. 技術面: 價量資料(price_volume)...
2. 籌碼面: 三大法人(foreign_trustment_investigator), 融資券(margin)...
3. 基本面: 營收(revenue), 損益表(statement_of_income)...

---

此外還有一份保存上市上櫃股票的列表(overview)，用來驗證收入資料是不是股票

---

目前已上線使用的資料有

1. overview
2. general/technical/price_volume
3. individual/technical/price_volume

---

agent和depository均以上述的方式命名建構，資料實際上是儲存在depository中，agent則是儲存負責對應名稱資料維護的程式碼(core.py)，該程式碼會被Database.py中的Conncetor使用。

## Database.py

**Connector** : 連結某類的資料，填入該資料的路徑，並以逗點分隔，則可執行該agent定義的function，使用方法如下:

```python
#連結至路徑的範例
Connector('overview')
Connector('general','technical','price_volume')
Connector('individual','technical','price_volume')
#執行該agent內容範例(以'individual','technical','price_volume'為例)
#在agent/individual/technical/price_volume/core.py中定義的function皆可使用
Connector('individual','technical','price_volume').update() #從最後一次更新的位置開始更新
Connector('individual','technical','price_volume').acquire('2021-09-27') #取得該日期資料
```

## Dataset.py

- Stock: 透過代號或名稱取得單一股票，特定屬性對應資料庫的內容，範例如下:

  ```python
  Stock(1101).ohlcv #取得代號1101股票的價量資料
  Stock('亞泥').ohlcv #取得亞泥的價量資料
  ```

- StockPile: 取得一批股票，透過疊代方式取得Stock物件

  1. ActiveStocks: 選取最後一筆更新資料日期還有price_volume資料的股票(還在市場上交易)
  2. SelectIDs: 選取一群股票代號
  3. SelectNames: 選取一群股票名稱
  4. Range: 依照代號順序選取
  5. All: 選取所有有過資料的股票，無論是否還在市場上交易

- General: 使用方式同Stock，但只接受taiex(上市), otc(上櫃)

## Structure.py

Compile Model的四個必要物件，透過繼承方式使用

1. Estimator: 從大盤資料計算資料給detector, trader使用

   ```python
   class MyEstimator(Estimator):
       def __init__(self):
           super().__init__(vars())
           #設定call所需的計算物件
       def call(self,resource):
           #resource.taiex取得上市指數資料
           #resource.otc取得上櫃指數資料
           return for_detector, for_trader #提供資料給detector, trader
   ```

2. Detector: 時間序列上True為列入候選標的時間，從該時間點開始持續追蹤股票資訊

   例如: [F,F,T,F,F,F,F,T,F,F]的第3, 8個時間點都是偵測到的時間點

   ```python
   class MyDetector(Detector):
       def __init__(self):
           super().__init__(vars())
           #設定call所需的計算物件
       def call(self,resource):
           #resource是Stock物件
           return conditions #回傳時間序列的布林訊號，True為偵測到的時間點
   ```

3. Trader

   ```python
   class MyTrader(Trader):
       def __init__(self,first_stage_success_rate,second_stage_success_rate,cut_rate):
           super().__init__(vars())
           #設定call所需的計算物件
       def call(self,resource):
           #計算cancel, buy, success, fail, flush所需的資料
       def buy(self,resource,record):
           return True,'reason' #(True:市價單 or numerical:限價單,str:備註)
       def cancel(self,resource,record):
           return None #True:取消追蹤 or None:無變更
       def success(self,resource,record):
           return True, 'reason' #(True:市價單 or numerical:限價單,str:備註)
       def fail(self,resource,record):
           return True, 'reason' #(True:市價單 or numerical:若大於收盤價則掛市價單,str:備註)
       def flush(self,resource,record):
           return True, 'reason' #(True:市價單 or numerical:限價單,str:備註)
   ```

4. Adjuster

   ```python
   class MyAdjuster(Adjuster):
       def __init__(self,valid_period=0,groupby_key=None):
           super().__init__(vars())
           
       def rate(self,keys,ref):
           if (ref.sell_price/ref.buy_price).mean() < 1:
               return 0
   ```

   

**Model**

## Analyser.py

## Resample.py

## Indicator.py, FloatingIndicator.py

## Signalizer.py

## Filter.py

---

# 待辦事項

