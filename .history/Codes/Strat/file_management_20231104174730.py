import os
import shutil
import pandas as pd
from typing import Union, Set, Any
import importlib.util


class FileManagement(object):
    def __init__(self, ceiling_directory: str = "DATAAAASSS"):
        self.ceiling_directory = ceiling_directory

    def search(self, target_name: str, start_path: str, search_type: str = 'both') -> Union[str, None]:
        visited = set()

        def dfs_search(current_path: str) -> Union[str, None]:
            if current_path in visited or (self.ceiling_directory and current_path.endswith(self.ceiling_directory)):
                return None

            visited.add(current_path)

            all_names = os.listdir(current_path)

            if target_name in all_names:
                full_path = os.path.join(current_path, target_name)

                if (search_type == 'both' or
                    (search_type == 'file' and os.path.isfile(full_path)) or
                    (search_type == 'folder' and os.path.isdir(full_path))):
                    return full_path

            for name in all_names:
                new_path = os.path.join(current_path, name)
                if os.path.isdir(new_path):
                    result = dfs_search(new_path)
                    if result:
                        return result

            parent_path = os.path.dirname(current_path)
            if parent_path != current_path:
                return dfs_search(parent_path)

            return None

        return dfs_search(start_path)

    def load_data(self, folder_name: str, file_name: str, **kwargs) -> pd.DataFrame:

        start_path = os.getcwd()

        folder_path = self.search(target_name=folder_name, start_path=start_path, search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"Folder {folder_name} not found.")

        file_path = os.path.join(folder_path, file_name)

        file_extension = os.path.splitext(file_name)[1]

        if file_extension == '.csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_extension == '.xlsx':
            data = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"File extension {file_extension} not supported.")

        print(f"Loaded file {file_name} from folder {folder_name}.")

        return data

    def save_data(self, folder_name: str, file_name: str, data: pd.DataFrame, **kwargs) -> None:

        start_path = os.getcwd()

        folder_path = self.search(target_name=folder_name, start_path=start_path, search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"Folder {folder_name} not found.")

        file_path = os.path.join(folder_path, file_name)

        file_extension = os.path.splitext(file_name)[1]

        if file_extension == '.csv':
            data.to_csv(file_path, **kwargs)
        elif file_extension == '.xlsx':
            data.to_excel(file_path, **kwargs)
        else:
            raise ValueError(f"File extension {file_extension} not supported.")

        print(f"Saved file {file_name} to folder {folder_name}.")



if __name__ == '__main__':
    fm = FileManagement()
    data = fm.load_data(folder_name='Data', file_name='sectors.csv')
    print(data.head())