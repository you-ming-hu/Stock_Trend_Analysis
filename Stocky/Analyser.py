import datetime
import numpy as np
import pandas as pd
import pathlib
import ipywidgets as widgets
from IPython.display import clear_output
import re
import warnings
import pickle
from collections import namedtuple
pd.options.display.max_columns = 40

warnings.filterwarnings('ignore',category=DeprecationWarning)

class SingleModel:
    def __init__(self,model):
        result = model.result
        config = str(model.config)
        buy_history = widgets.Button(description='追蹤股票')
        sell_history = widgets.Button(description='持股')
        all_history = widgets.Button(description='所有股票')
        cancel = widgets.Button(description='取消追蹤')
        history = widgets.HBox([widgets.Label('歷史資料'),buy_history,sell_history,all_history,cancel])
        
        buy  = widgets.Button(description='追蹤股票')
        sell = widgets.Button(description='持股')
        processing = widgets.HBox([widgets.Label('進行中股票'),buy,sell])
        
        self.__header = widgets.VBox([history,processing])
        self.__header_buttons = [buy_history,sell_history,all_history,cancel,buy,sell]

        def create_buy_history_pannel(btn):
            pannel = HistoryPannel('buy',self.merge(result.complete,result.buy),config)
            self.construct(pannel,btn)
        def create_sell_history_pannel(btn):
            pannel = HistoryPannel('sell',self.merge(result.complete,result.sell),config)
            self.construct(pannel,btn)
        def create_all_history_table(btn):
            pannel = HistoryPannel('all',result.complete,config)
            self.construct(pannel,btn)
        def create_cancel_pannel(btn):
            pannel = CancelPannel(result.cancel,config)
            self.construct(pannel,btn)
        def create_buy_pannel(btn):
            pannel = BuyPannel(result.buy,config)
            self.construct(pannel,btn)
        def create_sell_pannel(btn):
            pannel = SellPannel(result.sell,config)
            self.construct(pannel,btn)
        
        buy_history.on_click(create_buy_history_pannel)
        sell_history.on_click(create_sell_history_pannel)
        all_history.on_click(create_all_history_table)
        cancel.on_click(create_cancel_pannel)
        
        buy.on_click(create_buy_pannel)
        sell.on_click(create_sell_pannel)

        all_history.click()
        
    def merge(self,main,additional):
        additional = additional.copy()
        data = main.loc[main.ID.isin(additional.ID.unique())]

        additional['name(ID)'] = additional['name'] +'('+ additional['ID'] + ')'
        additional = additional.set_index('ID')

        data = data.reset_index().set_index('ID')
        data = data.join(additional.add_prefix('__'))
        data = data.reset_index().set_index('index')
        data = data.sort_index()
        return data
    
    def construct(self,pannel,btn):
        self.__core = pannel
        btn.icon = 'check'
        btn.button_style = 'success'
        for button in self.__header_buttons:
            if button is not btn:
                button.icon = ''
                button.button_style = ''
        clear_output(wait=True)
        display(self.__header)
        pannel.show_GUI()
    
    @classmethod
    def load(clz,file_path,drive='D'):
        file_path = pathlib.Path(file_path)
        if drive == 'D':
            file_path = pathlib.Path('D:\\','StockLab').joinpath(file_path)
        result = pickle.loads(file_path.joinpath('result.pkl').read_bytes())
        result_container = namedtuple('result',['cancel','buy','sell','complete'])
        result = result_container(**result)
        config = file_path.joinpath('config.txt').read_text()
        model_container = namedtuple('model',['result','config'])
        model = model_container(result=result,config=config)
        return clz(model)

class Pannel:
    def __init__(self,data,config):
        self.__settings = {'display_halt':False,'row':None,'column':None}
        self.__output = widgets.Output()
        
        self.__searchbox = self.create_searchbox(data)
        
        body = self.__searchbox.GUI
        model_config = self.create_config_tab(config)
        settings = self.create_settings()
        XQ_interface = self.create_XQ_interface()
        self.update_output_table(data)
        
        GUI = widgets.Accordion([body,model_config,settings,XQ_interface])
        GUI.set_title(0, '查詢股票')
        GUI.set_title(1, '檢視模型結構')
        GUI.set_title(2, '調整檢視')
        GUI.set_title(3, '匯出XQ自選股')
        self.GUI = GUI
        
    def show_GUI(self):
        display(self.GUI)
        display(self.__output)
        
    def create_config_tab(self,config):
        output = widgets.Output()
        with output:
            print(config)
        return output
        
    def create_settings(self):
        row_settings = self.row_settings()
        column_settings = self.column_settings()
        settings = widgets.Accordion([row_settings,column_settings])
        settings.set_title(0, '調整索引')
        settings.set_title(1, '調整欄位')
        return settings
        
    def row_settings(self):
        reset = widgets.Button(description='重置')
        start = widgets.BoundedIntText(min=0,max=100,value=0,description='起:')
        end = widgets.BoundedIntText(min=0,max=100,value=100,description='迄:')
        slide = widgets.IntRangeSlider(min=0,max=100,value=[0,100],step=1,description='起-迄',continuous_update=False,readout=True,layout = {'width':'90%'})
        count = widgets.Label()
        def reset_click(btn):
            slide.value = (slide.min,slide.max)
        def value_input(change):
            slide.value = [start.value,end.value]
        def slide_input(change):
            start.max = change.new[1]
            start.min = change.owner.min
            end.min = change.new[0]
            end.max = change.owner.max
            start.value = change.new[0]
            end.value = change.new[1]
            count.value = f'共選取{change.new[1] - change.new[0]:6} 列'
            self.display()
        reset.on_click(reset_click)
        start.observe(value_input,names='value')
        end.observe(value_input,names='value')
        slide.observe(slide_input,names='value')
        
        max_rows = widgets.IntText(value=pd.options.display.max_rows,description='變更顯示最多列數',style = {'description_width': 'initial'})
        def display_max_rows(change):
            pd.options.display.max_rows = change.new
            self.display()
        max_rows.observe(display_max_rows,names='value')
        
        row_settings = widgets.VBox([widgets.HBox([reset,start,end,count]),slide,max_rows])
        self.__settings.update({'row':slide})
        return row_settings
    
    def column_settings(self):
        select = widgets.SelectMultiple(description='選取顯示欄位:',rows=30,layout={'width':'35%'})
        reset = widgets.Button(description='全部選取')
        def select_change(change):
            self.display()
        def reset_click(btn):
            select.value = select.options
        select.observe(select_change,names='value')
        reset.on_click(reset_click)
        
        column_settings = widgets.HBox([select,reset])
        self.__settings.update({'column':select})
        return column_settings
        
    def update_output_table(self,table):
        self.__output_table = table[[c for c in table.columns if not c.startswith('__')]]
        self.__settings.update({'display_halt':True})
        self.update_row_settings()
        self.update_column_settings()
        self.__settings.update({'display_halt':False})
        self.display()
        
    def update_row_settings(self):
        slide = self.__settings['row']
        slide.max = self.__output_table.shape[0]
        slide.value = (0,slide.max)
    
    def update_column_settings(self):
        select = self.__settings['column']
        select.options = self.__output_table.columns.tolist()
        select.rows = min(30,len(select.options))
        select.value = select.options
        
    @property
    def display_table(self):
        row_setting = self.__settings['row'].value
        column_setting = list(self.__settings['column'].value)
        table = self.__output_table[column_setting].iloc[slice(*row_setting)]
        return table
    
    def display(self):
        with self.__output:
            if not self.__settings['display_halt']:
                clear_output(wait=True)
                display(self.display_table)
            
    @property            
    def search(self):
        return self.__searchbox.search
    
    def create_XQ_interface(self):
        split_by_date = widgets.Checkbox(value=True,description='依日期拆分')
        folder = widgets.Text(description='資料夾名稱:')
        button = widgets.Button(description='確定匯出')
        export = widgets.VBox([split_by_date,widgets.HBox([folder,button])])
        
        def to_XQ(btn):
            root = pathlib.Path(pathlib.Path.home(),'Desktop','Stocky')
            folder_name = datetime.datetime.today().strftime(f'%Y-%m-%d-%H-%M-%S({folder.value})')
            assert len(set(folder_name).intersection(set(r'\/:*?"><|'))) == 0,'名稱不能包含以下字元\\/:*?"><|'
            folder_path = root.joinpath(folder_name)
            folder_path.mkdir(parents=True)
            if split_by_date.value:
                for date, stocks in self.search.groupby('detect_date'):
                    text_name = datetime.datetime.strftime(date,'%Y-%m-%d.txt')
                    text_path = folder_path.joinpath(text_name)
                    text_path.write_text('\n'.join([f'{ID}.TW' for ID in stocks['ID']]))
            else:
                text_name = f'{folder.value}_compressed.txt'
                text_path = folder_path.joinpath(text_name)
                text_path.write_text('\n'.join([f'{ID}.TW' for ID in self.search['ID'].unique()]))
            self.update_output_table(self.search)
            with self.__output:
                print('匯出完畢',f'匯出路徑: {folder_path.as_posix()}')
            
        button.on_click(to_XQ)
        return export

class CancelPannel(Pannel):
    def __init__(self,table,config):
        super().__init__(table,config)
    def create_searchbox(self,data):
        sb = SearchBox(self,data)
        with sb.Page('日期查詢') as page:
            content = sb.HBox([
                sb.date('偵測日期','detect_date'),
                sb.date('取消日期','cancel_date'),
            ])
            page.add(content)
        with sb.Page('數值查詢') as page:
            content = sb.Tab({
                '價格':sb.VBox([
                    sb.value('偵測價格','detect_price'),
                    sb.value('取消價格','cancel_price'),
                ]),
                '其他':sb.VBox([
                    sb.value('追蹤天數','days')
                ])
            })
            page.add(content)
        with sb.Page('屬性查詢') as page:
            content = sb.Tab({
                '股票資訊':sb.VBox([
                    sb.category('股票代號','ID'),
                    sb.category('股票名稱','name'),
                    sb.category('產業別','industry'),
                    sb.category('市場別','market'),
                ]),
                '其他':sb.VBox([
                    sb.category('取消指示','cancel_instruct'),
                ])
            })
            page.add(content)
        return sb    

class BuyPannel(Pannel):
    def __init__(self,table,config):
        super().__init__(table,config)
    def create_searchbox(self,data):
        sb = SearchBox(self,data)
        with sb.Page('日期查詢') as page:
            content = sb.HBox([
                sb.date('偵測日期','detect_date'),
                sb.date('當前日期','current_date'),
            ])
            page.add(content)
        with sb.Page('數值查詢') as page:
            content = sb.Tab({
                '價格':sb.VBox([
                    sb.value('偵測價格','detect_price'),
                    sb.value('當前價格','current_price'),
                ]),
                '其他':sb.VBox([
                    sb.value('權重','weight'),
                    sb.value('調整','adjust'),
                    sb.value('調整權重','adjusted_weight'),
                    sb.value('追蹤天數','days')
                ])
            })
            page.add(content)
        with sb.Page('屬性查詢') as page:
            content = sb.Tab({
                '股票資訊':sb.VBox([
                    sb.category('股票代號','ID'),
                    sb.category('股票名稱','name'),
                    sb.category('產業別','industry'),
                    sb.category('市場別','market'),
                ]),
                '其他':sb.VBox([
                    sb.category('買單','buy_order'),
                    sb.category('購買指示','buy_instruct'),
                ])
            })
            page.add(content)
        return sb

class SellPannel(Pannel):
    def __init__(self,table,config):
        super().__init__(table,config)
    def create_searchbox(self,data):
        sb = SearchBox(self,data)
        with sb.Page('日期查詢') as page:
            content = sb.HBox([
                sb.date('偵測日期','detect_date'),
                sb.date('購買日期','buy_date'),
                sb.date('當前日期','current_date'),
            ])
            page.add(content)
        with sb.Page('數值查詢') as page:
            content = sb.Tab({
                '報酬率':sb.VBox([
                    sb.value('浮動報酬率','floating_return'),
                    sb.value('稅後浮動報酬率','floating_return_after_tax')
                ]),
                '價格':sb.VBox([
                    sb.value('偵測價格','detect_price'),
                    sb.value('購買價格','buy_price'),
                    sb.value('當前價格','current_price'),
                ]),
                '其他':sb.VBox([
                    sb.value('權重','weight'),
                    sb.value('調整','adjust'),
                    sb.value('調整權重','adjusted_weight'),
                    sb.value('追蹤天數','days')
                ])
            })
            page.add(content)
        with sb.Page('屬性查詢') as page:
            content = sb.Tab({
                '股票資訊':sb.VBox([
                    sb.category('股票代號','ID'),
                    sb.category('股票名稱','name'),
                    sb.category('產業別','industry'),
                    sb.category('市場別','market'),
                ]),
                '收益率':sb.Accordion({
                    '浮動':sb.VBox([
                        sb.category('稅前','floating_revenue'),
                        sb.category('稅後','floating_revenue_after_tax'),
                    ]),
                '其他':sb.VBox([
                    sb.category('購買指示','buy_instruct'),
                    sb.category('賣單','sell_order'),
                    sb.category('賣出原因','sell_reason'),
                    sb.category('賣出指示','sell_instruct'),
                    sb.category('註記','note')
                    ])
                })
            })
            page.add(content)
        with sb.Page('聚合查詢') as page:
            content = sb.aggregate('sell')
            page.add(content)
        
        with sb.Page('統計介面') as page:
            content = sb.Statistics()
            page.add(content)
        return sb

class HistoryPannel(Pannel):
    def __init__(self,kind,table,config):
        self.__kind = kind
        super().__init__(table,config)
        
    def create_searchbox(self,data):
        sb = SearchBox(self,data)
        with sb.Page('日期查詢') as page:
            content = sb.HBox([
                sb.date('偵測日期','detect_date'),
                sb.date('購買日期','buy_date'),
                sb.date('賣出日期','sell_date'),
                sb.date('目前日期','current_date'),
            ])
            page.add(content)
            
        with sb.Page('數值查詢') as page:
            content = sb.Tab({
                '報酬率':sb.VBox([
                    sb.value('已實現報酬率','realized_return'),
                    sb.value('稅後已實現報酬率','realized_return_after_tax'),
                    sb.value('浮動報酬率','floating_return'),
                    sb.value('稅後浮動報酬率','floating_return_after_tax')
                ]),
                '價格':sb.VBox([
                    sb.value('偵測價格','detect_price'),
                    sb.value('購買價格','buy_price'),
                    sb.value('賣出價格','sell_price'),
                    sb.value('當前價格','current_price')
                ]),
                '其他':sb.VBox([
                    sb.value('權重','weight'),
                    sb.value('調整','adjust'),
                    sb.value('調整權重','adjusted_weight'),
                    sb.value('追蹤天數','days')
                ])
            })
            page.add(content)
        
        with sb.Page('屬性查詢') as page:
            content = sb.Tab({
                '股票資訊':sb.VBox([
                    sb.category('股票代號','ID'),
                    sb.category('股票名稱','name'),
                    sb.category('產業別','industry'),
                    sb.category('市場別','market'),
                ]),
                '收益率':sb.Accordion({
                    '以實現':sb.VBox([
                        sb.category('稅前','realized_revenue'),
                        sb.category('稅後','realized_revenue_after_tax'),
                    ]),
                    '浮動':sb.VBox([
                        sb.category('稅前','floating_revenue'),
                        sb.category('稅後','floating_revenue_after_tax'),
                    ])
                }),
                '其他':sb.VBox([
                    sb.category('購買指示','buy_instruct'),
                    sb.category('賣出原因','sell_reason'),
                    sb.category('賣出指示','sell_instruct'),
                    sb.category('註記','note'),
                ])
            })
            page.add(content)
            
        with sb.Page('聚合查詢') as page:
            content = sb.aggregate('history')
            page.add(content)
        
        if self.__kind == 'buy':
            with sb.Page('追蹤股票特別欄位') as page:
                content = sb.Tab({
                    '股票資訊':sb.HBox([
                        sb.multiselect('名稱(代號)','__name(ID)'),
                        sb.multiselect('產業別','__industry'),
                        sb.multiselect('市場別','__market')
                    ]),
                    '偵測資訊':sb.HBox([
                        sb.multiselect('偵測日期','__detect_date'),
                        sb.multiselect('偵測價格','__detect_price')
                    ]),
                    '購買資訊':sb.HBox([
                        sb.multiselect('買單','__buy_order'),
                        sb.multiselect('購買指示','__buy_instruct')
                    ]),
                    '權重資訊':sb.HBox([
                        sb.multiselect('權重','__weight'),
                        sb.multiselect('調整','__adjust'),
                        sb.multiselect('調整權重','__adjusted_weight')
                    ]),
                    '其他':sb.HBox([
                        sb.multiselect('追蹤天數','__days'),
                    ]),
                })
                page.add(content)

        elif self.__kind == 'sell':
            with sb.Page('持股特別欄位') as page:
                content = sb.Tab({
                    '股票資訊':sb.HBox([
                        sb.multiselect('名稱(代號)','__name(ID)'),
                        sb.multiselect('產業別','__industry'),
                        sb.multiselect('市場別','__market')
                    ]),
                    '偵測資訊':sb.HBox([
                        sb.multiselect('偵測日期','__detect_date'),
                        sb.multiselect('偵測價格','__detect_price')
                    ]),
                    '購買資訊':sb.HBox([
                        sb.multiselect('購買日期','__buy_date'),
                        sb.multiselect('購買價格','__buy_price'),
                        sb.multiselect('購買指示','__buy_instruct')
                    ]),
                    '權重資訊':sb.HBox([
                        sb.multiselect('權重','__weight'),
                        sb.multiselect('調整','__adjust'),
                        sb.multiselect('調整權重','__adjusted_weight')
                    ]),
                    '賣出資訊':sb.HBox([
                        sb.multiselect('賣單','__sell_order'),
                        sb.multiselect('賣出指示','__sell_instruct')
                    ]),
                    '浮動報酬率':sb.HBox([
                        sb.multiselect('稅前','__floating_return'),
                        sb.multiselect('稅後','__floating_return_after_tax')
                    ]),
                    '浮動損益':sb.HBox([
                        sb.multiselect('稅前','__floating_revenue'),
                        sb.multiselect('稅後','__floating_revenue_after_tax')
                    ]),
                    '其他':sb.HBox([
                        sb.multiselect('追蹤天數','__days'),
                        sb.multiselect('註記','__note'),
                    ]),
                })
                page.add(content)

        with sb.Page('統計介面') as page:
            content = sb.Statistics()
            page.add(content)
        return sb

class SearchBox:
    def __init__(self,pannel,data):
        self.update_output_table = pannel.update_output_table
        self.__pages = {}
        self.__indexes = {}
        self.__cache = data.index
        self.__data = data
        self.__options = {}
        self.__multi_options = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        self.__pages.update({self.page_title:self.page_content})
        del self.page_title
        del self.page_content
    
    def Page(self,title):
        self.page_title = title
        return self
    
    def add(self,content):
        self.page_content = content
        
    def Container(self,box,children):
        content = box(children)
        clear = self.clear(content)
        return widgets.VBox([content,clear],layout={'border': '1px solid black'})
    
    def MultiBlockContainer(self,box,children):
        box = self.Container(box,list(children.values()))
        content = box.children[0]
        [content.set_title(i,name) for i,name in enumerate(children.keys())]
        return box
        
    def HBox(self,children):
        return self.Container(widgets.HBox,children)
    
    def VBox(self,children):
        return self.Container(widgets.VBox,children)
    
    def Tab(self,children):
        return self.MultiBlockContainer(widgets.Tab,children)
    
    def Accordion(self,children):
        return self.MultiBlockContainer(widgets.Accordion,children)
        
    @property
    def GUI(self):
        gui = widgets.Tab(list(self.__pages.values()))
        [gui.set_title(i,title) for i,title in enumerate(self.__pages.keys())]
        clear = self.clear(gui)
        gui = widgets.VBox([gui,clear])
        self.update_statistics()
        return gui
    
    def Statistics(self):
        self.__statistics = widgets.Output()
        return self.__statistics

    def date(self,name,column):
        col = self.__data[column]
        index = self.__data.index
        self.__indexes.update({column:index})
        
        layout={'width':'90%'}
        
        title = widgets.Label(f'{name}({column})'.replace('__',''))
        select = widgets.Text(description='指定日期',layout=layout)
        range_label = widgets.Label('日期區間')
        start = widgets.Text(description='起:',layout=layout)
        end = widgets.Text(description='迄:',layout=layout)
        count_label = widgets.Label('')
        input_cell = widgets.VBox([select,range_label,start,end,count_label])
        clear = self.clear(input_cell)
        
        def date_format(change):
            x = re.match(r'(\d{0,4})-?(\d{0,2})-?(\d{0,2})',change.new)
            s = x.group(1)
            if x.group(2) != '':
                s = s + '-' + x.group(2)
            if x.group(3) != '':
                s = s + '-' + x.group(3)
            change.owner.value = s
        
        def parse_input():
            warnings.filterwarnings('ignore',category=FutureWarning)
            try:
                state = (select.value=='',start.value=='',end.value=='')
                if state == (True,True,True):
                    count_label.value=''
                    new_index = index
                elif state == (False,True,True):
                    count_label.value=''
                    new_index = col.reset_index().set_index(column).loc[select.value].set_index('index').index
                elif state == (True,False,True):
                    count_label.value=''
                    new_index = col.reset_index().set_index(column).loc[start.value:].set_index('index').index
                elif state == (True,True,False):
                    count_label.value=''
                    new_index = col.reset_index().set_index(column).loc[:end.value].set_index('index').index
                elif state == (True,False,False):
                    diff = (pd.Timestamp(end.value) - pd.Timestamp(start.value)).days
                    count_label.value=f'共{diff:5}天(含休假日)'
                    new_index = col.reset_index().set_index(column).loc[start.value:end.value].set_index('index').index
                else:
                    new_index = None
                    
                if new_index is not None:
                    self.update_index_and_display(column,new_index)
            except (KeyError, ValueError):
                pass
            warnings.resetwarnings()
            
        def fill_select(change):
            if change.new != '':
                start.value = ''
                end.value = ''
                date_format(change)
            parse_input()

        def fill_range(change):
            if change.new != '':
                select.value = ''
                date_format(change)
            parse_input()
        
        select.observe(fill_select,names='value')
        start.observe(fill_range,names='value')
        end.observe(fill_range,names='value')
        
        cell = widgets.VBox([title,input_cell,clear],layout={'border': '1px solid black'})
        return cell
        
    def value(self,name,column):
        col = self.__data[column]
        index = self.__data.index
        self.__indexes.update({column:index})
            
        layout = {'width':'20%'}
        
        title = widgets.Label(f'{name}({column})'.replace('__',''),layout={'width':'60%'})
        by_percentage = widgets.ToggleButton(value=False,description='依百分位數',layout=layout)
        select = widgets.Text(description='指定數值',layout=layout)
        range_label = widgets.Label('數值區間',layout=layout)
        minimum = widgets.Text(description='大於等於:',layout=layout)
        maximum = widgets.Text(description='小於等於:',layout=layout)
        input_cell = widgets.HBox([by_percentage,select,range_label,minimum,maximum])
        clear = self.clear(input_cell)
        
        
        def number_format(change):
            change.owner.value = re.sub(r'[^0-9\-.]','',change.new)
        
        def parse_input():
            try:
                state = (select.value=='',minimum.value=='',maximum.value=='')
                if state == (True,True,True):
                    new_index = index
                elif state == (False,True,True):
                    s = float(select.value)
                    if by_percentage.value:
                        s = col.quantile(s/100,interpolation='nearest')
                    new_index = index[col == s]
                elif state == (True,False,True):
                    mini = float(minimum.value)
                    if by_percentage.value:
                        mini = col.quantile(mini/100)
                    new_index = index[col >= mini]
                elif state == (True,True,False):
                    maxi = float(maximum.value)
                    if by_percentage.value:
                        maxi = col.quantile(maxi/100)
                    new_index = index[col <= maxi]
                elif state == (True,False,False):
                    maxi = float(maximum.value)
                    mini = float(minimum.value)
                    if by_percentage.value:
                        mini = col.quantile(mini/100)
                        maxi = col.quantile(maxi/100)
                    if maxi >= mini:
                        new_index = index[(col >= mini)&(col <= maxi)]
                    else:
                        new_index = index[(col >= mini)|(col <= maxi)]
                else:
                    new_index = None
                    
                if new_index is not None:
                    self.update_index_and_display(column,new_index)
            except ValueError:
                pass
            
        def fill_select(change):
            if change.new != '':
                minimum.value = ''
                maximum.value = ''
                number_format(change)
            parse_input()
                
        def fill_range(change):
            if change.new != '':
                select.value = ''
                number_format(change)
            parse_input()
            
        by_percentage.observe(lambda change: parse_input(),names='value')
        select.observe(fill_select,names='value')
        minimum.observe(fill_range,names='value')
        maximum.observe(fill_range,names='value')
        
        cell = widgets.HBox([title,by_percentage,select,range_label,minimum,maximum,clear],layout={'border': '1px solid black'})
        return cell
    
    def category(self,name,column):
        col = self.__data[column]
        index = self.__data.index
        self.__indexes.update({column:index})
        
        title = widgets.Label(f'{name}({column})'.replace('__',''),layout={'width':'60%'})
        option = widgets.Combobox(options=['']+sorted(str(s) for s in col.unique()),ensure_option=True)
        clear = self.clear(widgets.Box([option]))
        
        self.__options.update({column:option})
        
        def select_option(change):
            if change.new != '':
                try:
                    value = eval(change.new)
                except NameError:
                    value = change.new
                new_index = index[col == value]
            else:
                new_index = index
            self.update_index_and_display(column,new_index)
            
        option.observe(select_option,names='value')
        cell = widgets.HBox([title,option,clear],layout={'border': '1px solid black'})
        return cell
    
    def multiselect(self,name,column):
        col = self.__data[column]
        index = self.__data.index
        self.__indexes.update({column:index})
        
        title = widgets.Label(f'{name}({column})'.replace('__',''),layout={'width':'60%'})
        options = sorted(col.unique())
        select = widgets.SelectMultiple(options=options,value=options,rows=20)
        reset = widgets.Button(description='重置')
        clear = self.clear(widgets.Box([select]))
        
        self.__multi_options.update({column:select})
        
        def select_func(change):
            new_index = index[col.isin(change.new)]
            self.update_index_and_display(column,new_index)
        
        def reset_func(btn):
            for idx_name in self.__multi_options.keys():
                if idx_name != column:
                    self.update_index(idx_name,index)
            for idx_name,opt in self.__multi_options.items():
                if idx_name != column:
                    new_options = sorted(self.__data.loc[self.__cache,idx_name].unique())
                    opt.value = new_options
        
        select.observe(select_func,names='value')
        reset.on_click(reset_func)
        cell = widgets.VBox([title,select,reset,clear],layout={'border': '1px solid black'})
        return cell
    
    def aggregate(self,kind):
        agg = Aggregate(self)
        agg.build(kind)
        return agg.GUI
    
    def clear(self,box):
        def clear_click(btn):
            for c in box.children:
                if isinstance(c,widgets.Text):
                    c.value = ''
                elif isinstance(c,widgets.SelectMultiple):
                    c.value = c.options
                elif isinstance(c,widgets.ToggleButton):
                    c.value = False
                elif isinstance(c,widgets.Box):
                    for i in c.children:
                        if isinstance(i,widgets.Button):
                            if i.description == '清除':
                                i.click()
        button = widgets.Button(description='清除')
        button.on_click(clear_click)
        return button
    
    def update_index_and_display(self,name,new_index):
        self.update_index(name,new_index)
        self.update_output_table(self.search)
        
    def update_index(self,name,new_index):
        self.__indexes.update({name:new_index})
        self.update_cache()
        self.update_options()
        self.update_statistics()
    
    @property
    def search(self):
        return self.__data.loc[self.__cache]
        
    def update_cache(self):
        self.__cache = sorted(set.intersection(*[set(s) for s in self.__indexes.values()]))
    
    def update_options(self):
        for column,opt in self.__options.items():
            new_options = ['']+sorted(str(s) for s in self.__data.loc[self.__cache,column].unique())
            opt.options = new_options
            
    def calculate_statistics(self,data):
        warnings.filterwarnings('ignore',category=RuntimeWarning)
        def calculate_return_statistics(table,column):
            try:
                weight = table['weight']
                adjusted = table['adjusted_weight']
                col = table[column]

                mean = col.mean()
                weighted_mean = (col*weight).sum()/weight.sum()
                adjusted_mean = (col*adjusted).sum()/adjusted.sum()
                geo_mean = 2**np.log2(col).mean()
                geo_std = 2**np.log2(col).std()
            except KeyError:
                mean = np.nan
                weighted_mean = np.nan
                adjusted_mean = np.nan
                geo_mean = np.nan
                geo_std = np.nan

            return [mean,weighted_mean,adjusted_mean,geo_mean,geo_std]

        def get_table_return_statistics(table):
            realized_return = calculate_return_statistics(table,'realized_return')
            realized_return_after_tax = calculate_return_statistics(table,'realized_return_after_tax')

            floating_return = calculate_return_statistics(table,'floating_return')
            floating_return_after_tax = calculate_return_statistics(table,'floating_return_after_tax')

            before_tax = []
            before_tax.extend(realized_return)
            before_tax.extend(floating_return)

            after_tax = []
            after_tax.extend(realized_return_after_tax)
            after_tax.extend(floating_return_after_tax)

            column = pd.MultiIndex.from_product([['報酬率'],
                                    ['已實現','未實現'],
                                    ['算數平均','加權平均','調整加權平均','幾何平均','幾何標準差']])

            output = pd.DataFrame([before_tax,after_tax],index=['稅前','稅後'],columns=column)
            return output
        
        def calculate_revenue_statistics(table,column):
            try:
                col = table[column]
                gain = (col=='gain').mean()
                tie = (col=='tie').mean()
                loss = (col=='loss').mean()
            except KeyError:
                gain = np.nan
                tie = np.nan
                loss = np.nan
            return [gain,tie,loss]

        def get_table_revenue_statistics(table):
            realized_revenue = calculate_revenue_statistics(table,'realized_revenue')
            realized_revenue_after_tax = calculate_revenue_statistics(table,'realized_revenue_after_tax')

            floating_revenue = calculate_revenue_statistics(table,'floating_revenue')
            floating_revenue_after_tax = calculate_revenue_statistics(table,'floating_revenue_after_tax')

            before_tax = []
            before_tax.extend(realized_revenue)
            before_tax.extend(floating_revenue)

            after_tax = []
            after_tax.extend(realized_revenue_after_tax)
            after_tax.extend(floating_revenue_after_tax)

            column = pd.MultiIndex.from_product([['收益率'],
                                    ['已實現','未實現'],
                                    ['收益','無','損失']])

            output = pd.DataFrame([before_tax,after_tax],index=['稅前','稅後'],columns=column)
            return output

        def calculate_sell_reason_statistics(table):
            try:
                col = table['sell_reason']
                success = (col=='success').mean()
                flush = (col=='flush').mean()
                fail = (col=='fail').mean()
            except KeyError:
                success = np.nan
                flush = np.nan
                fail = np.nan
            return pd.DataFrame({'成功率':[success],'出清率':[flush],'失敗率':[fail]},index=[''])
        
        count = np.nan if data.empty else data.shape[0]
        count = pd.DataFrame({'資料筆數':[count]},index=[''])
        return_statistics = get_table_return_statistics(data)
        revenue_statistics = get_table_revenue_statistics(data)
        reason_statistics = calculate_sell_reason_statistics(data)
        warnings.resetwarnings()
        return count,return_statistics,revenue_statistics,reason_statistics
        
    def update_statistics(self):
        try:
            statistics_page = self.__statistics
            count_, return_, revenue_, reason_ = self.calculate_statistics(self.search)
            with statistics_page:
                clear_output(wait=True)
                display(count_)
                display(return_)
                display(revenue_)
                display(reason_)
        except AttributeError:
            pass

class Aggregate:
    def __init__(self,searchbox):
        self.__groupby_buttons = []
        self.__data = searchbox._SearchBox__data
        self.__searchbox = searchbox
        self.__searchbox._SearchBox__indexes.update({'aggregate':self.__data.index})
        self.__groupby = {'category':[],'numerical':[],'date':[]}
        self.__choose = {'apply_scope':None,
                         'target':None,
                         'min_value':None,
                         'max_value':None,
                         'head_or_tail':None,
                         'nth':None}
        
    @property
    def search(self):
        table = self.__searchbox.search.copy()
        table['name(ID)'] = table['name'] + '(' + table['ID'] + ')'
        return table
    
    @property
    def GUI(self):
        gui = widgets.Tab([self.groupby_tab,self.search_tab])
        gui.set_title(0,'聚合選項')
        gui.set_title(1,'聚合資料查詢')
        display_detail = widgets.Button(description='顯示原始資料')
        clear = widgets.Button(description='清除')
        
        def display_detail_click(btn):
            self.__searchbox.update_output_table(self.__searchbox.search)
        def clear_click(btn):
            gui.children[0].children[0].children[1].children[0].click()
        
        display_detail.on_click(display_detail_click)
        clear.on_click(clear_click)
        
        gui = widgets.VBox([gui,clear,display_detail])
        return gui
        
    def build(self,kind):
        if kind == 'history':
            self.build_history()
        elif kind == 'sell':
            self.build_sell()
        self.ChooseTab()
        
    def build_history(self):
        self.GroupbyTab(
            self.Blocks({
                '股票資訊':self.HBox([
                    self.category('市場別','market'),
                    self.category('產業別','industry'),
                    self.category('名稱(代號)','name(ID)')
                ]),
                '偵測資訊':self.VBox([
                    self.date('偵測日期','detect_date'),
                    self.numerical('偵測價格','detect_price')
                ]),
                '購買資訊':self.VBox([
                    self.date('購買日期','buy_date'),
                    self.numerical('購買價格','buy_price'),
                    self.category('購買指示','buy_instruct')
                ]),
                '權重資訊':self.HBox([
                    self.category('權重','weight'),
                    self.category('調整','adjust'),
                    self.category('調整權重','adjusted_weight')
                ]),
                '賣出資訊':self.VBox([
                    self.date('賣出日期','sell_date'),
                    self.numerical('賣出價格','sell_price'),
                    self.numerical('報酬率','realized_return'),
                    self.numerical('稅後報酬率','realized_return_after_tax'),
                    self.category('收益率','realized_revenue'),
                    self.category('稅後收益率','realized_revenue_after_tax')
                ]),
                '其他資訊':self.VBox([
                    self.numerical('追蹤天數','days'),
                    self.category('註記','note')
                ])
            })
        )

    def build_sell(self):
        self.GroupbyTab(
            self.Blocks({
                '股票資訊':self.HBox([
                    self.category('市場別','market'),
                    self.category('產業別','industry'),
                    self.category('名稱(代號)','name(ID)')
                ]),
                '偵測資訊':self.VBox([
                    self.date('偵測日期','detect_date'),
                    self.numerical('偵測價格','detect_price')
                ]),
                '購買資訊':self.VBox([
                    self.date('購買日期','buy_date'),
                    self.numerical('購買價格','buy_price'),
                    self.category('購買指示','buy_instruct')
                ]),
                '權重資訊':self.HBox([
                    self.category('權重','weight'),
                    self.category('調整','adjust'),
                    self.category('調整權重','adjusted_weight')
                ]),
                '賣出資訊':self.VBox([
                    self.category('賣單','sell_order'),
                    self.category('賣出原因','sell_reason'),
                    self.category('賣出指示','sell_instruct')
                ]),
                '其他資訊':self.VBox([
                    self.numerical('追蹤天數','days'),
                    self.category('註記','note')
                ])
            })
        )
        
    def GroupbyTab(self,content):
        clear = widgets.Button(description='清除')
        confirm = widgets.Button(description='開始聚合運算')
        def clear_click(btn):
            try:
                del self.__aggregate_table
            except AttributeError:
                pass
            self.__searchbox.update_index('aggregate',self.__data.index)
            for button in self.__groupby_buttons:
                if button.value:
                    button.value = False
            self.clear_choose_tab()
        def confirm_click(btn):
            self.groupby()
        clear.on_click(clear_click)
        confirm.on_click(confirm_click)
        self.groupby_tab = widgets.VBox([content,widgets.HBox([clear,confirm])])
        
    def ChooseTab(self):
        apply_scope = widgets.SelectMultiple(rows=20)
        self.__choose.update({'apply_scope':apply_scope})
        apply_scope_box = widgets.VBox([widgets.Label('適用範圍:'),apply_scope],layout={'border': '1px solid black'})
        
        target = widgets.Select(rows=20)
        self.__choose.update({'target':target})
        target_box = widgets.VBox([widgets.Label('搜索對象:'),target],layout={'border': '1px solid black'})

        min_value = widgets.Text(description='大於等於')
        self.__choose.update({'min_value':min_value})
        max_value = widgets.Text(description='小於等於')
        self.__choose.update({'max_value':max_value})
        value_range = widgets.HBox([min_value,max_value])
        value_range = widgets.VBox([widgets.Label('指定數值區間'),value_range],layout={'border': '1px solid black'})

        head_or_tail = widgets.ToggleButtons(description='選取: ',options=['前','後'])
        self.__choose.update({'head_or_tail':head_or_tail})
        nth = widgets.BoundedIntText(value=3,min=0,max=10000,layout={'width':'20%'})
        self.__choose.update({'nth':nth})
        choose = widgets.HBox([head_or_tail,nth])
        choose = widgets.VBox([widgets.Label('選取前或後指定筆資料'),choose],layout={'border': '1px solid black'})

        clear = widgets.Button(description='清除')
        confirm = widgets.Button(description='開始查詢')
        
        def clear_click(btn):
            self.__searchbox.update_index('aggregate',self.__data.index)
            apply_scope.value=[]
            target.value=None
            min_value.value=''
            max_value.value=''
            head_or_tail.value = head_or_tail.options[0]
            nth.value = 3
            
        def confirm_click(btn):
            by = list(apply_scope.value)
            on = target.value
            table = self.__aggregate_table.reset_index()
            min_v = float(min_value.value) if min_value.value != '' else -np.inf
            max_v = float(max_value.value) if max_value.value != '' else np.inf
            ascending = True if head_or_tail.value == '後' else False
            count = nth.value
            
            table = table.loc[(table[on] >= min_v) & (table[on] <= max_v)]
            apply_func = lambda df: df.sort_values(on,ascending=ascending).iloc[:count]
            if len(by) != 0:
                table = table.groupby(by).apply(apply_func)
            else:
                table = table.sort_values(on,ascending=ascending).iloc[:count]
            try:
                table = table.set_index(list(apply_scope.options))
                self.__searchbox.update_index('aggregate',set(table['index'].explode()))
                self.__searchbox.update_output_table(table.drop(['index'],axis=1))
            except KeyError:
                self.__searchbox.update_index('aggregate',set())
                self.__searchbox.update_output_table(table)
            
        clear.on_click(clear_click)
        confirm.on_click(confirm_click)

        self.search_tab = widgets.VBox([
            widgets.HBox([apply_scope_box,target_box]),
            value_range,
            choose,
            widgets.HBox([clear,confirm])])
        
    def update_choose_tab(self):
        self.__choose['apply_scope'].options = list(self.__aggregate_table.index.names)
        self.__choose['target'].options = list(self.__aggregate_table.drop(['index'],axis=1).columns)
    
    def clear_choose_tab(self):
        self.__choose['apply_scope'].options = []
        self.__choose['target'].options = []
        self.__choose['min_value'].value = ''
        self.__choose['max_value'].value = ''
        self.__choose['head_or_tail'].value = self.__choose['head_or_tail'].options[0]
        self.__choose['nth'].value = 3
        
    def Blocks(self,children):
        return widgets.VBox(
            list(widgets.VBox([widgets.Label(title),block]) for title,block in children.items()),
            layout = {'border': '1px solid black'}
        )
    def HBox(self,children):
        return widgets.HBox(children)
    
    def VBox(self,children):
        return widgets.VBox(children)
    
    def button(self,name):
        button = widgets.ToggleButton(value=False,description=name)
        self.__groupby_buttons.append(button)
        def check(change):
            if change.new:
                change.owner.icon = 'check'
                change.owner.button_style='success'
            else:
                change.owner.icon = ''
                change.owner.button_style=''
        button.observe(check,names='value')
        return button

    def category(self,name,column):
        button = self.button(name)
        self.__groupby['category'].append({'button':button,'column':column})
        return button
    
    def value(self,name):
        button = self.button(name)
        granularity = widgets.Text(disabled=True,continuous_update=False,layout={'width':'80%'})
        def activate(change):
            if change.new:
                granularity.disabled=False
            else:
                granularity.disabled=True
                granularity.value=''
        button.observe(activate,names='value')
        return button,granularity
    
    def numerical(self,name,column):
        button,granularity = self.value(name)
        granularity.description = '數值間距:'
        granularity.placeholder = '填入區間或是間格大小(數字)，例如:[0,100,200,1000,10000]或是10'
        self.__groupby['numerical'].append({'button':button,'column':column,'granularity':granularity})  
        return widgets.HBox([button,granularity])
    
    def date(self,name,column):
        button,granularity = self.value(name)
        granularity.description = '時間頻率:'
        granularity.placeholder = '數字+[Y年,M月,W周,D日]擇一，例如:3M'
        self.__groupby['date'].append({'button':button,'column':column,'granularity':granularity})  
        return widgets.HBox([button,granularity])
            
    def groupby(self):
        self.__searchbox.update_index('aggregate',self.__data.index)
        group = []
        for g in self.__groupby['category']:
            if g['button'].value:
                group.append(g['column'])
                
        for g in self.__groupby['numerical']:
            if g['button'].value:
                granularity = eval(g['granularity'].value)
                if isinstance(granularity,int):
                    grouper = pd.cut(self.search[g['column']].astype(float),range(0,10000,granularity))
                elif isinstance(granularity,float):
                    grouper = pd.cut(self.search[g['column']].astype(float),np.arange(0,100,granularity))
                else:
                    grouper = pd.cut(self.search[g['column']].astype(float),granularity)
                group.append(grouper)
        
        for g in self.__groupby['date']:
            if g['button'].value:
                grouper = pd.Grouper(key=g['column'],freq = g['granularity'].value)
                group.append(grouper)
        self.__aggregate_table = self.search.groupby(group,observed=True).apply(self.apply_func).dropna(how='all')
        self.__searchbox.update_output_table(self.__aggregate_table.drop(['index'],axis=1))
        self.update_choose_tab()
        
    def apply_func(self,data):
        count_, return_, revenue_, reason_ = self.__searchbox.calculate_statistics(data)
        def flat(df):
            output = df.unstack()
            output.index  = ['_'.join(x) for x in output.index]
            output = output.to_frame().T
            output.index=['']
            return output
        return_ = flat(return_)
        revenue_ = flat(revenue_)
        output = pd.concat([count_, return_, revenue_, reason_],axis=1)
        output['index'] = np.nan if data.empty else (data.index.tolist(),)
        output = output.squeeze()
        return output
    
class MultiModel:
    def __init__(self,project_name,drive='D'):
        self.data = pd.DataFrame()
        read_path = pathlib.Path(project_name)
        if drive == 'D':
            read_path = pathlib.Path('D:\\','StockLab').joinpath(read_path)
        for s in read_path.iterdir():
            self.data = self.data.append(pickle.loads(s.joinpath('summary.pkl').read_bytes()),ignore_index=True)