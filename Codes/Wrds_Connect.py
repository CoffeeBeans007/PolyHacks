import wrds
import time
class WRDSDataFetcher:
    # Defining object
    def __init__(self, library_name, username, password):
        self.library_name = library_name
        self.username = username
        self.password = password
        self.conn = None
    # Connect to SQL server
    def connect(self):
        try:
            self.conn = wrds.Connection(wrds_username=self.username)
            print("Connected to WRDS successfully")
        except wrds.ConnectionError as e:
            print(f"Connection error: {e}")

    # Return the dataset  from a given library name and a given dataset_name
    def get_dataset(self, dataset_name):
        if not self.conn:
            print("Not connected to WRDS")
            return None
        
        try:
            table_info = self.conn.get_table(library=self.library_name, table=dataset_name)
            print(f"Structure of the table '{dataset_name}' in library '{self.library_name}':")
            print(table_info.head())  
            return table_info
        except wrds.ConnectionError as e:
            print(f"Error fetching dataset: {e}")
            return None

    # Close Program
    def close(self):
        if self.conn:
            self.conn.close()
            print("Connection to WRDS closed.")

class NameCompiler:

    def createQuery(dataset:str, year:int, month:str, day:int):
        month=time.strptime(month,'%b').tm_mon
        return f"{dataset}_{year}{month:02}{day:02}"

if __name__ == "__main__":
    
    wrds_username = 'Secrets'
    wrds_password = 'Secrets'


    library_name = 'taqm_2021'  # Change to the desired library
    
    wrds_fetcher = WRDSDataFetcher(library_name, wrds_username, wrds_password)
    wrds_fetcher.connect()
    
    
    dataset_name = NameCompiler.createQuery("complete_nbbo",2021,'Jun',"04")

   
    dataset = wrds_fetcher.get_dataset(dataset_name)
    print(dataset)
    
    # Close the connection when done
    wrds_fetcher.close()