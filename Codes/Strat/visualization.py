from os_helper import OsHelper
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_sector_weights_over_time(df: pd.DataFrame, title: str = "Sector Weights Over Time") -> plt.Figure:
    """
    Plots the weights of sectors over time, ensuring that the X-axis displays the correct dates.

    Args:
    df (pd.DataFrame): The DataFrame containing the weights.
    title (str): The title of the plot.

    Returns:
    plt.Figure: The matplotlib figure object for further customization or saving.
    """
    # Ensure the DataFrame index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Normalize the weights for plotting percentages
    df_norm = df.div(df.sum(axis=1), axis=0)

    # Prepare the plot
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")

    # Plot the data
    for column in df_norm.columns:
        plt.plot(df_norm.index, df_norm[column], label=column)

    # Formatting the x-axis to show years only
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Rotate dates for better readability
    plt.xticks(rotation=45)

    # Additional formatting
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Weight Percentage')
    plt.legend(title='Sectors', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Set a background color to the plot area
    plt.gca().set_facecolor('#eaeaf2')

    # Return the figure object
    return plt.gcf()

def plot_sector_cumulative_returns(df: pd.DataFrame, title: str = "Sector Cumulative Returns Over Time") -> plt.Figure:
    """
    Plots the cumulative returns of sectors over time, ensuring that the X-axis displays the correct dates.

    Args:
        df (pd.DataFrame): The DataFrame containing the cumulative returns.
        title (str): The title of the plot.

    Returns:
        plt.Figure: The matplotlib figure object for further customization or saving.
    """
    # Ensure the DataFrame index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Prepare the plot
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")
    # use log scale
    plt.yscale('log')

    # Plot the data
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    # Formatting the x-axis to show dates formatted by year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Rotate dates for better readability
    plt.xticks(rotation=45)

    # Additional formatting
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return (log scale)')
    plt.legend(title='Sectors', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Set a background color to the plot area
    plt.gca().set_facecolor('#eaeaf2')

    # Return the figure object
    return plt.gcf()


def save_figure(fig: plt.Figure, folder_path: str, file_name: str):
    """
    Saves the figure to the specified folder with the given filename.

    Args:
    fig (plt.Figure): The figure object to save.
    folder_path (str): The relative or absolute path to the folder.
    file_name (str): The name of the file to save the figure as.
    """
    # Construct the full path
    full_path = f"{folder_path}/{file_name}"

    # Save the figure
    fig.savefig(full_path, bbox_inches='tight')

    print(f"Graph saved to {full_path}")


if __name__ == "__main__":
    os_helper = OsHelper()

    sectors_drifted_weights = os_helper.read_data(directory_name='final data', file_name='sectors_drifted_weights.csv', index_col=0)
    sectors_compounded_returns = os_helper.read_data(directory_name='final data', file_name='sectors_compounded_returns.csv', index_col=0)

    print(sectors_drifted_weights.head())
    print(sectors_compounded_returns.head())

    fig = plot_sector_weights_over_time(df=sectors_drifted_weights, title="Sectors Weights Over Time")
    save_figure(fig=fig, folder_path='../Graphs', file_name='sector_weights_over_time.png')

    # Plotting the sector cumulative returns
    fig_returns = plot_sector_cumulative_returns(df=sectors_compounded_returns, title="Sector Cumulative Returns Over Time")
    save_figure(fig=fig_returns, folder_path='../Graphs', file_name='sector_cumulative_returns_over_time.png')


