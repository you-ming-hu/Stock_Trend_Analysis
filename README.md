# Stock Trend Analysis

專案名稱: Stocky 

投資人傾向對自己的決策過度自信，造成錯誤的判斷。為了避免主觀認知的偏差，量化概 念的介入的是必要的。本專案分為主要三個部分: 1.資料蒐集與存取, 2.建立模型的框架, 3.實際交易輔助。透過建立rule-based model，在資料集上進行回測，並取得詳細交易紀 錄，分析模型的優劣。完成的模型也容易部屬直接使用，提供每日掛單的建議。

## 專案結構

```
├─ Demo : 專案使用範例
|  ├─ DataUpdateManager.ipynb 	透過Stocky內dataset相關的API，進行資料維護
|  └─ Research					使用的範例
|     ├─ Analyse.ipynb			分析模型回測結果
|     ├─ R0.ipynb				建立第0號模型，進行回測
|     └─ ...					建立第n號模型，進行回測
├─ Stocky						主程式與資料存放處
|  └─ ...
└─ 下載html					建立新資料時，收集較大量歷史資料用
   └─ ...
```

註記:

1. 下載html

   建立初始建立資料時需要收集大量的歷史資料，向證交所網站請求資料時有一定的機率會取得錯誤的資料，或是IP被阻擋，且同一資料不同時期的內容也可能不同，所以還要額外整理才能夠合併成一份資料。在容量可接受的情況下，處理以上問題最直接的辦法是把請求到的html儲存在本地端，之後再全部重新請求驗證第一次儲存的內容是否有差異，確定沒有錯誤的內容後才開始解析html，把內容格式統一，整理在一起才放進資料庫。

