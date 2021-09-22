import runpy
import pathlib
import pandas as pd

class Connector:
    
    def __init__(self,subject,*category):
        database = pathlib.Path(__file__).parent.joinpath('Database')
        depository = database.joinpath('depository')
        agent = database.joinpath('agent')
        
        subject = pathlib.Path(subject)
        category = pathlib.Path(*category)
        
        overview = pd.read_csv(depository.joinpath('overview','data.csv').as_posix(),index_col=0)
        
        self.core = agent.joinpath(subject,category,'core.py')
        self.functions = runpy.run_path(self.core.as_posix(),
                                        dict(agent=agent.joinpath(subject,category),
                                             depository=depository.joinpath(subject),
                                             category=category,
                                             overview=overview))
        
        
        
    def __getattr__(self,key):
        print(self.core)
        return self.functions[key]