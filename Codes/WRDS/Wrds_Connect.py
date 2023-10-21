import time
import wrds
from typing import Optional, Union
from pandas import DataFrame
import pandas as pd


class WRDSDataFetcher:
    """
    A class to fetch data from WRDS.

    Parameters
    ----------
    library_name : str
        The name of the WRDS library.
    username : str
        The WRDS username.
    password : str
        The WRDS password.

    Attributes
    ----------
    conn : wrds.Connection, optional
        The WRDS connection object.
    """

    def __init__(self, library_name: str, username: str) -> None:
        self.library_name = library_name
        self.username = username
        self.conn = None

    def connect(self) -> None:
        """
        Connect to WRDS.
        """
        try:
            self.conn = wrds.Connection(wrds_username=self.username)
            print("Connected to WRDS successfully.")
        except wrds.ConnectionError as e:
            print(f"Connection error: {e}")

    def get_dataset(self, dataset_name: str) -> Optional[DataFrame]:
        """
        Fetches a dataset from WRDS.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to fetch.

        Returns
        -------
        DataFrame or None
            The fetched dataset, or None if fetching fails.
        """
        if not self.conn:
            print("Not connected to WRDS.")
            return None

        try:
            table_info = self.conn.get_table(library=self.library_name, table=dataset_name)
            print(f"Structure of the table '{dataset_name}' in library '{self.library_name}':")
            print(table_info.head())
            return table_info
        except wrds.ConnectionError as e:
            print(f"Error fetching dataset: {e}")
            return None

    def close(self) -> None:
        """
        Close the WRDS connection.
        """
        if self.conn:
            self.conn.close()
            print("Connection to WRDS closed.")

    def fundaDataFetch(self, comp_cusip: str):
        ccomrt=self.conn.raw_sql(

            f"""
            select ITEM5601, ITEM7011, ITEM7210, ITEM7220, ITEM7230, ITEM7240, ITEM7250, ITEM8101, ITEM8106, ITEM8111, ITEM8121, ITEM8136, ITEM8226, ITEM8231, ITEM8236, ITEM8306, ITEM8316, ITEM8336, ITEM8366, ITEM8371, ITEM8401, ITEM8406, ITEM8601, ITEM8611, ITEM8621, ITEM8626, ITEM8631, ITEM8636, ITEM6004

            from trws.wrds_ws_funda

            where item6004='{comp_cusip}' 
            """)
        ccomrt.fillna(method='bfill', inplace=True)
        ccomrt_first_row=ccomrt.iloc[len(ccomrt)-2] if not ccomrt.empty else None
        print(ccomrt_first_row)


class NameCompiler:
    """
    A class to compile query names.
    """

    @staticmethod
    def create_query(dataset: str, year: int, month: str, day: int) -> str:
        """
        Create a query based on dataset, year, month, and day.

        Parameters
        ----------
        dataset : str
            Dataset name.
        year : int
            Year.
        month : str
            Month in string format.
        day : int
            Day.

        Returns
        -------
        str
            The generated query.
        """
        month = time.strptime(month, '%b').tm_mon
        return f"{dataset}_{year}{month:02}{day:02}"

    



if __name__ == "__main__":
    wrds_username = 'Secrets'
    wrds_password = 'Secrets'
    wrds_username='' # test, second test..  # Change to the desired library
    comp_cusip='037833100'
    wrds_fetcher=WRDSDataFetcher(wrds_username,wrds_password)
    wrds_fetcher.connect()
    wrds_fetcher.fundaDataFetch(comp_cusip)
    wrds_fetcher.close()
