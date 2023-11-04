import pandas as pd

class data_input():
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

    
if __name__ == "__main__":
    data = data_input('Data/low.csv')
    print(data.get_all_tickers())