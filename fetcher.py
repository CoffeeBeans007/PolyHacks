import pandas as pd

class data_input:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)
        self.data_name = self.path.split('/')[-1].split('.')[0]
        
    def get_data(self, ticker):
        return self.data[[self.data_name+ "_"+ ticker]]
    
    def get_all_tickers(self):
        uncleaned_columns = self.data.columns
        cleaned_columns = []
        for column in uncleaned_columns:
            cleaned_columns.append(column.split('_')[-1])
        cleaned_columns.pop(0)
        return cleaned_columns
    
    def get_data(self,):
        return self.data

class merged_data():
    def __init__(self) -> None:
        self.data_names = ['adjusted_close', 'close', 'high', 'low', 'open', 'volume', 'dividend', 'split']
        self.data = pd.DataFrame()
    def fetch_and_merge(self,):
        for data_name in self.data_names:
            data = data_input('Data/'+data_name+'.csv').get_data()
            self.data = pd.concat([self.data, data], axis=1)
            
if __name__ == "__main__":
    data = merged_data()
    data.fetch_and_merge()
    data.data.to_csv('Data/merged_data.csv')