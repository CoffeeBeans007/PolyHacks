import pandas as pd
from os_helper import OsHelper


def prepare_drifted_weights_format_submission(drifted_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare drifted weights for submission.

    Args:
        drifted_weights (pd.DataFrame): DataFrame containing drifted weights.

    Returns:
        pd.DataFrame: DataFrame containing drifted weights in submission format.
    """

    # Rename ticker columns by removing ' US Equity' and prefixing with 'weight_'
    drifted_weights.columns = drifted_weights.columns.str.replace(' US Equity', '')
    drifted_weights.columns = [f"weight_{ticker}" for ticker in drifted_weights.columns]

    # Reset the index and rename the 'date' column and index
    drifted_weights.reset_index(inplace=True)
    drifted_weights.rename(columns={'index': 'date'}, inplace=True)
    drifted_weights.index.rename('id', inplace=True)

    return drifted_weights

# Example usage:
# Assuming 'drifted_weights_df' is a DataFrame that contains drifted weights
# with columns named like 'AAPL US Equity', 'MSFT US Equity', etc.

#



if __name__ == '__main__':
    os_helper = OsHelper()
    drifted_weights = os_helper.read_data(directory_name='final data', file_name='drifted_weights.csv', index_col=0)
    submission_sample = os_helper.read_data(directory_name='base data', file_name='sample_submission.csv', index_col=0)
    print(drifted_weights.head())
    print(submission_sample.head())

    submission_weights = prepare_drifted_weights_format_submission(drifted_weights=drifted_weights)

    print(submission_weights.head())

    os_helper.write_data(directory_name='final data', file_name='submission_weights.csv', data_frame=submission_weights)

