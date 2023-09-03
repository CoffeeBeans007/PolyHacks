import time
import wrds
from typing import Optional, Union
from pandas import DataFrame


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

    def __init__(self, library_name: str, username: str, password: str) -> None:
        self.library_name = library_name
        self.username = username
        self.password = password
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
    wrds_password = 'Secrets' # test, second test..
    library_name = 'taqm_2021'  # Change to the desired library

    wrds_fetcher = WRDSDataFetcher(library_name, wrds_username, wrds_password)
    wrds_fetcher.connect()

    dataset_name = NameCompiler.create_query("complete_nbbo", 2021, 'Jun', 4)

    dataset = wrds_fetcher.get_dataset(dataset_name)
    print(dataset)

    wrds_fetcher.close()
