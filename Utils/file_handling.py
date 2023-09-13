import pandas as pd
from typing import Optional
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import os
import shutil
import glob
from zipfile import ZipFile


class FileHandler:

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

    @staticmethod
    def list_files(directory: str, extension: str) -> None:
        """
        Liste tous les fichiers d'une extension donnée dans un répertoire.

        Args:
            directory (str): Le chemin du répertoire à examiner.
            extension (str): L'extension des fichiers à lister.

        Returns:
            None
        """
        files = glob.glob(f"{directory}/*.{extension}")
        print(f"Files in directory: {files}")

    @staticmethod
    def delete_file(file_path: str) -> None:
        """
        Supprime un fichier spécifié.

        Args:
            file_path (str): Le chemin du fichier à supprimer.

        Returns:
            None
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print("The file does not exist.")

    @staticmethod
    def rename_file(old_name: str, new_name: str) -> None:
        """
        Renomme un fichier.

        Args:
            old_name (str): Le nom actuel du fichier.
            new_name (str): Le nouveau nom du fichier.

        Returns:
            None
        """
        os.rename(old_name, new_name)
        print(f"File has been renamed from {old_name} to {new_name}")

    @staticmethod
    def move_file(file_path: str, target_path: str) -> None:
        """
        Déplace un fichier vers un autre répertoire.

        Args:
            file_path (str): Le chemin du fichier à déplacer.
            target_path (str): Le répertoire de destination.

        Returns:
            None
        """
        shutil.move(file_path, target_path)
        print(f"File has been moved to {target_path}")

    @staticmethod
    def copy_file(file_path: str, target_path: str) -> None:
        """
        Copie un fichier vers un autre répertoire.

        Args:
            file_path (str): Le chemin du fichier à copier.
            target_path (str): Le répertoire de destination.

        Returns:
            None
        """
        shutil.copy(file_path, target_path)
        print(f"File has been copied to {target_path}")

    @staticmethod
    def compress_files(file_paths: list, zip_name: str) -> None:
        """
        Compresse plusieurs fichiers dans une archive zip.

        Args:
            file_paths (list): Liste des chemins des fichiers à compresser.
            zip_name (str): Le nom de l'archive zip résultante.

        Returns:
            None
        """
        with ZipFile(zip_name, 'w') as zipf:
            for file in file_paths:
                zipf.write(file)
        print(f"Files have been compressed into {zip_name}")


if __name__ == '__main__':
    print('This is file_handling.py')

    csv_path = "../Data/taq_20.TAQ_SP_500_2020_1sec.csv"
    parquet_path = "../Data/taq_20.TAQ_SP_500_2020_1sec.parquet"

    n_rows = 100_000
    file_name = f'../Data/taq_20.TAQ_SP_500_2020_1sec_{n_rows}.parquet'

    file_handler = FileHandler()

    # Use this line only once to convert the csv file to parquet file
    # data_preprocessing.convert_csv_to_parquet(csv_path=csv_path, parquet_path=parquet_path)
    file_handler.read_n_rows_from_parquet(parquet_path=parquet_path, n_rows=n_rows, file_name=file_name)








