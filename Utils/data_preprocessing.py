import pandas as pd
from typing import Optional
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class DataPreprocessing:

    def __init__(self):
        pass

    @staticmethod
    def convert_csv_to_parquet(csv_path: str, parquet_path: Optional[str] = None) -> None:
        """
            Convertit un fichier CSV en fichier Parquet.

            Args:
                csv_path (str): Chemin vers le fichier CSV à convertir.
                parquet_path (str, optional): Chemin de sortie du fichier Parquet.
                                              Si None, le même chemin que le CSV est utilisé avec l'extension .parquet.

            Returns:
                None
            """
        # Lecture du fichier CSV
        df = pd.read_csv(csv_path)

        # Si le chemin du fichier Parquet n'est pas spécifié, on utilise le même chemin que le fichier CSV avec l'extension .parquet
        if parquet_path is None:
            parquet_path = csv_path.rsplit('.', 1)[0] + '.parquet'

        # Écriture du fichier Parquet
        df.to_parquet(parquet_path, engine='pyarrow')

        print(f'Fichier CSV converti en Parquet avec succès : {parquet_path}')

        return None

    @staticmethod
    def read_n_rows_from_parquet(parquet_path: str, n_rows: int, file_name: str) -> None:
        """
        Read n_rows from a Parquet file and save to another Parquet file using PyArrow's iter_batches.

        Args:
            parquet_path (str): Path to the original Parquet file.
            n_rows (int): Number of rows to read from the original Parquet file.
            file_name (str): The name of the new Parquet file to save the n_rows to.

        Returns:
            None
        """
        # Initialize ParquetFile object
        pf = ParquetFile(parquet_path)

        # Read the first n_rows using iter_batches
        first_n_rows = next(pf.iter_batches(batch_size=n_rows))

        # Convert to DataFrame
        df = pa.Table.from_batches([first_n_rows]).to_pandas()

        # Save to a new Parquet file
        df.to_parquet(file_name, engine='pyarrow')

        print(f'Successfully read {n_rows} rows from {parquet_path} and saved to {file_name}')

        return None


if __name__ == '__main__':
    print('This is data_preprocessing.py')

    csv_path = "../Data/taq_20.TAQ_SP_500_2020_1sec.csv"
    parquet_path = "../Data/taq_20.TAQ_SP_500_2020_1sec.parquet"

    n_rows = 100_000
    file_name = f'../Data/taq_20.TAQ_SP_500_2020_1sec_{n_rows}.parquet'

    data_preprocessing = DataPreprocessing()

    # Use this line only once to convert the csv file to parquet file
    # data_preprocessing.convert_csv_to_parquet(csv_path=csv_path, parquet_path=parquet_path)
    data_preprocessing.read_n_rows_from_parquet(parquet_path=parquet_path, n_rows=n_rows, file_name=file_name)








