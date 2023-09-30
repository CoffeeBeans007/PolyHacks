import os
import re
import pandas as pd
from typing import Optional


class OHLCDataManager:
    """
    Class to manage OHLC (Open-High-Low-Close) data in finance.
    """

    def __init__(self, path_to_data: str):
        """
        Initialize OHLCDataManager with file path.
        """
        self.path_to_data = path_to_data
        self.data_extension = os.path.splitext(self.path_to_data)[1][1:]
        self.data = self.load_data()

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load data from the given file path.
        """
        try:
            if self.data_extension == 'csv':
                data = pd.read_csv(self.path_to_data, index_col=0)
            elif self.data_extension == 'xlsx':
                data = pd.read_excel(self.path_to_data, index_col=0)
            else:
                print("Unsupported file extension.")
                return None

            if not self.validate_columns(data):
                print("Data does not contain required OHLC columns.")
                return None

            if not self.validate_index(data):
                print("Data index is not in datetime format.")
                return None

            data.index = pd.to_datetime(data.index)
            self.infer_frequency(data)

            return data

        except FileNotFoundError:
            print(f"File {self.path_to_data} not found.")
            return None

    def validate_columns(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains the required OHLC columns.

        :param data: Dataframe to be validated.
        :return: True if valid, False otherwise.
        """
        required_columns = ['open', 'high', 'low', 'close']
        present_columns = [re.sub(r'[^a-zA-Z]', '', col.lower()) for col in data.columns]

        return all(col in present_columns for col in required_columns)

    def validate_index(self, data: pd.DataFrame) -> bool:
        """
        Validate that the index of the data is in datetime format.
        If not, attempt to convert it.

        :param data: Dataframe to be validated.
        :return: True if valid or successfully converted, False otherwise.
        """
        if pd.api.types.is_datetime64_any_dtype(data.index):
            return True
        else:
            try:
                data.index = pd.to_datetime(data.index)
                return True
            except Exception as e:
                print(f"Could not convert index to datetime format: {e}")
                return False

    def infer_frequency(self, data: pd.DataFrame):
        """
        Infer the frequency of the data and print it.

        :param data: Dataframe whose frequency is to be inferred.
        """
        time_diffs = data.index.to_series().diff()
        mode_freq = time_diffs.mode()[0]

        if pd.notna(mode_freq):
            # Convert the mode frequency to seconds for easier calculation
            mode_seconds = mode_freq.total_seconds()

            if mode_seconds >= 86400 and mode_seconds % 86400 == 0:
                unit = 'day(s)'
                value = mode_seconds // 86400
            elif mode_seconds >= 3600 and mode_seconds % 3600 == 0:
                unit = 'hour(s)'
                value = mode_seconds // 3600
            elif mode_seconds >= 60 and mode_seconds % 60 == 0:
                unit = 'minute(s)'
                value = mode_seconds // 60
            else:
                unit = 'second(s)'
                value = mode_seconds

            print(f"Inferred frequency of data is: {int(value)} {unit}")
        else:
            print("Could not infer the frequency of the data.")

    def save_data(self, path_to_save: str = None, file_extension: str = None):
        """
        Save the OHLC data to a specified path.

        :param path_to_save: The file path to save the data.
            If not provided, use the original path_to_data attribute.
        :param file_extension: The file extension for saving the data.
            If not provided, use the original data_extension attribute.
        """
        if path_to_save is None:
            path_to_save = self.path_to_data

        if file_extension is None:
            file_extension = self.data_extension

        # Determine the complete path to save the data
        complete_path = f"{path_to_save}.{file_extension}"

        try:
            if file_extension == 'csv':
                self.data.to_csv(complete_path)
            elif file_extension == 'xlsx':
                self.data.to_excel(complete_path)
            else:
                print(f"Unsupported file extension: {file_extension}. Data not saved.")
                return

            print(f"Data successfully saved to {complete_path}.")

        except Exception as e:
            print(f"An error occurred while saving the data: {e}")


if __name__ == '__main__':

    path_to_data = '../../Data/DC_Data/EURUSD_H1.csv'

    ohlc_data_manager = OHLCDataManager(path_to_data=path_to_data)
    print(ohlc_data_manager.data)
