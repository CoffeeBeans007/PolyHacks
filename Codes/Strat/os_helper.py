import os
import pandas as pd
import shutil


class OsHelper(object):
    def __init__(self, root_directory: str = "PolyFinance"):
        self.root_directory = root_directory

    def _is_valid_path(self, current_path: str, explored_paths: set) -> bool:
        return current_path not in explored_paths and \
            (not self.root_directory or not current_path.endswith(self.root_directory))

    def _recursive_search(self, current_path: str, target_name: str, explored_paths: set) -> str:
        explored_paths.add(current_path)

        available_names = os.listdir(current_path)

        if target_name in available_names:
            full_path = os.path.join(current_path, target_name)
            if os.path.isdir(full_path):
                return full_path

        for name in available_names:
            new_path = os.path.join(current_path, name)
            if os.path.isdir(new_path) and self._is_valid_path(new_path, explored_paths):
                result = self._recursive_search(new_path, target_name, explored_paths)
                if result:
                    return result

        parent_path = os.path.dirname(current_path)
        if parent_path != current_path and self._is_valid_path(parent_path, explored_paths):
            return self._recursive_search(parent_path, target_name, explored_paths)

        return None

    def find_folder_path(self, target_name: str, search_path: str) -> str:
        return self._recursive_search(search_path, target_name, set())

    def read_data(self, directory_name: str, file_name: str, **kwargs) -> pd.DataFrame:
        starting_path = os.getcwd()
        directory_path = self.find_folder_path(target_name=directory_name, search_path=starting_path)
        if directory_path is None:
            raise FileNotFoundError(f"Folder {directory_name} not found.")
        file_path = os.path.join(directory_path, file_name)
        return self._load_data(file_path, **kwargs)

    def _load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.csv':
            data_frame = pd.read_csv(file_path, **kwargs)
        elif file_extension == '.xlsx':
            data_frame = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"File extension {file_extension} not supported.")
        print(f"Loaded file {os.path.basename(file_path)} from folder {os.path.dirname(file_path)}.")
        return data_frame

    def write_data(self, directory_name: str, file_name: str, data_frame: pd.DataFrame, **kwargs) -> None:
        starting_path = os.getcwd()
        directory_path = self.find_folder_path(target_name=directory_name, search_path=starting_path)
        if directory_path is None:
            raise FileNotFoundError(f"Folder {directory_name} not found.")
        file_path = os.path.join(directory_path, file_name)
        self._save_data(file_path, data_frame, **kwargs)

    def _save_data(self, file_path: str, data_frame: pd.DataFrame, **kwargs) -> None:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.csv':
            data_frame.to_csv(file_path, **kwargs)
        elif file_extension == '.xlsx':
            data_frame.to_excel(file_path, **kwargs)
        else:
            raise ValueError(f"File extension {file_extension} not supported.")
        print(f"Saved file {os.path.basename(file_path)} to folder {os.path.dirname(file_path)}.")

    def move_file(self, src_file_name: str, dest_directory_name: str) -> None:
        starting_path = os.getcwd()
        dest_directory_path = self.find_folder_path(target_name=dest_directory_name, search_path=starting_path)
        if dest_directory_path is None:
            raise FileNotFoundError(f"Folder {dest_directory_name} not found.")

        src_file_path = os.path.join(starting_path, src_file_name)
        dest_file_path = os.path.join(dest_directory_path, src_file_name)

        shutil.move(src_file_path, dest_file_path)
        print(f"Moved file {src_file_name} to folder {dest_directory_path}.")


if __name__ == '__main__':
    file_manager = OsHelper()
    data_frame = file_manager.read_data(directory_name='Data', file_name='sectors.csv')
    print(data_frame.head())
