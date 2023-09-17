import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.lines as mlines
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)


class DirectionalChange:
    '''
    DC class:
    Capable of reading a pd.DataFrame object, transforming it into DC series,
    tagged with ongoing event(up_DC,down_DC,up_OS,down_OS).
    Moreover, it can also
    plot the DC upon the TS views with self.plotDC, with any given threshold;
    calculate DC-based indicators, like aTMV,Coastline,NDC,aR with any given threshold;
    '''

    def __init__(self, data):
        '''
        data: pd.DataFrame of open/high/close/low time series data.
        '''
        self.data = data

    def dc_pattern(self, threshold):
        '''
        Directional Change transformation
        '''
        # Initiation
        event_dict = [True]
        state_dict = ['up_OS']
        ph = self.data['Close'][0]
        ph_idx = 0
        pl = self.data['Close'][0]
        pl_idx = 0

        # Recording iteration
        for idx in range(1, len(self.data)):
            # Iterating the TS data

            if event_dict[-1]:
                # If the last event flag is True: uptrend confirmed

                if self.data['Close'][idx] <= (ph * (1 - threshold)):
                    # And if current close value is less than (last hightest record*(1-threshold))

                    if state_dict[-1] != 'up_DC':
                        # Also if the last state if not upward DC event, the predefined OS event flag should be reset.

                        for i in range(ph_idx + 1, len(event_dict)):
                            event_dict[i] = False
                            state_dict[i] = "down_DC"

                    # The event flag is flipped from True to False, a downward DC event is recorded
                    state_dict.append('down_DC')
                    event_dict.append(False)

                    # Store the current close value and TS index as the lowest record.
                    pl = self.data['Close'][idx]
                    pl_idx = idx

                else:
                    # No DC event founded yet, the last event flag should pass into current one.

                    if state_dict[-1] == 'up_DC':
                        # if last state is upward DC event, current state should be changed into upward OS event.
                        state_dict.append('up_OS')
                    else:
                        # if last state is upward OS event, continue
                        state_dict.append(state_dict[-1])
                    # Continue with last event flag
                    event_dict.append(event_dict[-1])

                    if ph < self.data['Close'][idx]:
                        # if current close value is higher than record, update the record
                        ph = self.data['Close'][idx]
                        ph_idx = idx

            else:
                # Same as above
                if self.data['Close'][idx] >= (pl * (1 + threshold)):

                    if state_dict[-1] != 'down_DC':

                        for i in range(pl_idx + 1, len(event_dict)):
                            event_dict[i] = True
                            state_dict[i] = "up_DC"

                    event_dict.append(True)
                    state_dict.append('up_DC')

                    ph = self.data['Close'][idx]
                    ph_idx = idx

                else:

                    if state_dict[-1] == 'down_DC':
                        state_dict.append('down_OS')
                    else:
                        state_dict.append(state_dict[-1])

                    event_dict.append(event_dict[-1])

                    if pl > self.data['Close'][idx]:
                        pl = self.data['Close'][idx]
                        pl_idx = idx

        event_dict = list(zip(event_dict, state_dict))
        return pd.DataFrame(event_dict, index=[self.data.index],
                            columns=['DC:Threshold{}'.format(threshold), 'State:Threshold{}'.format(threshold)])  # Dict

    def plotDC(self, threshold: float, plot: bool = True) -> None:
        """
        Visualizing DC upon TS data.

        Parameters:
        - threshold: float
            Threshold value for DC pattern.
        - plot: bool, optional (default=True)
            Flag to control whether to plot the graph.
        """
        df = self.dc_pattern(threshold)
        df['ts'] = list(self.data['Close'])
        df['shift'] = df.iloc[:, 1].shift(-1)

        initial_row = df.iloc[0, :]
        listPlot = [df.index[0][0], initial_row['ts']]
        legend_handles = []
        added_labels = set()

        if plot:
            plt.figure(figsize=(25, 12))
            plt.title('DC Patterns vs Time Series Data', fontsize=20)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)

        self.listExtreme = [listPlot]
        x_all = df['ts']

        if plot:
            plt.plot(x_all.index.map(lambda x: x[0]), x_all, 'cornflowerblue', label='Time Series Data')

        color_map = {
            ('up_OS', 'down_DC'): 'green',
            ('down_DC', 'down_OS'): 'orange',
            ('down_OS', 'up_DC'): 'red',
            ('up_DC', 'up_OS'): 'lightgreen',
            ('up_DC', 'down_DC'): 'black',
            ('down_DC', 'up_DC'): 'blue'
        }

        label_map = {
            'green': 'Up OS to Down DC',
            'orange': 'Down DC to Down OS',
            'red': 'Down OS to Up DC',
            'lightgreen': 'Up DC to Up OS',
            'black': 'Up DC to Down DC',
            'blue': 'Down DC to Up DC'
        }

        for idx, row in df.iterrows():
            if idx == 0:
                continue

            current_color = color_map.get((row[1], row[-1]), None)
            if current_color:
                if row[1] in {'up_OS', 'down_OS', 'up_DC', 'down_DC'}:
                    self.listExtreme.append([idx[0], row['ts']])

                if plot:
                    plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], current_color)

                    if current_color not in added_labels:
                        legend_handles.append(
                            mlines.Line2D([], [], color=current_color, label=label_map[current_color]))
                        added_labels.add(current_color)

                listPlot = [idx[0], row['ts']]

        if plot:
            plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'lightgreen')
            legend_handles.append(mlines.Line2D([], [], color='cornflowerblue', label='Time Series Data'))
            plt.legend(handles=legend_handles, fontsize='large')
            plt.show()

        self.listExtreme.append([idx[0], row['ts']])

    def calculateATMV(self, threshold: float) -> List[Tuple[Tuple, float]]:
        """
        Calculate the Absolute Theoretical Movement Value (ATMV) for each directional change.

        Parameters:
        - threshold: float
            Threshold value for identifying directional changes.

        Returns:
        - List[Tuple[Tuple, float]]
            A list containing pairs of date ranges and their corresponding ATMV.
        """
        aTMV = []
        self.plotDC(threshold, plot=False)
        previous = self.listExtreme[0]

        for turningPoint in self.listExtreme[1:]:
            currentATMV = np.abs((turningPoint[1] - previous[1]) / previous[1] / threshold)
            aTMV.append([[previous[0], turningPoint[0]], currentATMV])
            previous = turningPoint

        return aTMV

    def calculateAR(self, threshold: float) -> List[Tuple[Tuple, float]]:
        """
        Calculate the Adjusted Return (AR) for each directional change.

        Parameters:
        - threshold: float
            Threshold value for identifying directional changes.

        Returns:
        - List[Tuple[Tuple, float]]
            A list containing pairs of date ranges and their corresponding AR.
        """
        aTMV = self.calculateATMV(threshold)
        aR = list(map(lambda x: x[1] / ((x[0][1] - x[0][0]).days), aTMV))
        timeSpan = list(zip(*aTMV))[0]
        aR = list(zip(timeSpan, aR))

        return aR

    def calculateCoastline(self, threshold: float) -> float:
        """
        Calculate the total 'coastline', which is the sum of all ATMVs.

        Parameters:
        - threshold: float
            Threshold value for identifying directional changes.

        Returns:
        - float
            The sum of all ATMVs.
        """
        aTMV = self.calculateATMV(threshold)
        Coastline = sum(list(zip(*aTMV))[1])

        return Coastline

    def calculateNDC(self, threshold: float) -> int:
        """
        Calculate the total number of directional changes.

        Parameters:
        - threshold: float
            Threshold value for identifying directional changes.

        Returns:
        - int
            The total number of directional changes.
        """
        return len(self.calculateATMV(threshold))


def prepare_data_dict(path: str) -> Dict[str, pd.DataFrame]:
    """
    Prépare un dictionnaire de DataFrames à partir d'un fichier Excel.

    Cette fonction prend en entrée le chemin d'un fichier Excel et retourne un dictionnaire de DataFrames.
    Chaque DataFrame est associé à une feuille du fichier Excel.

    Parameters
    ----------
    path : str
        Le chemin du fichier Excel à lire.

    Returns
    -------
    dict[str, pd.DataFrame]
        Un dictionnaire où chaque clé est le nom d'un index et la valeur associée est un DataFrame contenant les données de cet index.

    Examples
    --------
    # prepare_data_dict("path/to/excel/file.xlsx")
    {'Index1': DataFrame_object1, 'Index2': DataFrame_object2, ...}

    """

    # Lire les noms des feuilles dans le fichier Excel
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names

    # Lire la feuille "Indices" pour obtenir les noms des index
    df_idx = pd.read_excel(xls, sheet_name="Indices", index_col=0)

    # Initialiser le dictionnaire pour stocker les DataFrames
    data_dict = dict()

    # Parcourir les index et remplir le dictionnaire avec les DataFrames associés
    for idx in df_idx["Index"]:
        if idx in sheet_names:
            data_dict[idx] = pd.read_excel(xls, sheet_name=idx, index_col=0)
            data_dict[idx].index = pd.to_datetime(data_dict[idx].index, format="%Y%m%d")
        else:
            print(f"La feuille pour l'index {idx} n'a pas été trouvée dans le fichier Excel.")

    # Fermer le fichier Excel
    xls.close()

    return data_dict


def convert_to_dataframe(atmv_list, ar_list):
    # Format time spans as string
    atmv_list = [[f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", val] for (start, end), val in
                 atmv_list]
    ar_list = [[f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", val] for (start, end), val in ar_list]

    # Convert ATMV list to DataFrame
    atmv_df = pd.DataFrame(atmv_list, columns=['Time Span', 'ATMV'])

    # Convert AR list to DataFrame
    ar_df = pd.DataFrame(ar_list, columns=['Time Span', 'AR'])

    return atmv_df, ar_df

# Main Streamlit App
def main():
    st.title('Directional Change Analyzer')

    path = "../../Data/DC_data/Index_daily_close_2019_2023.xlsx"
    data_dict = prepare_data_dict(path)

    # Select the index
    index_choice = st.selectbox(
        'Select Index:',
        list(data_dict.keys())
    )

    # Select the threshold
    threshold = st.slider('Set Threshold:', min_value=0.0, max_value=0.2, value=0.01, step=0.001)

    # Perform calculations
    df_selected = data_dict[index_choice][['Close']]
    dc = DirectionalChange(df_selected)
    df = dc.dc_pattern(threshold=threshold)

    # Display DataFrame
    st.write('## Data Table')
    st.dataframe(df)

    # Display Visualization
    st.write('## Visualization')
    dc.plotDC(threshold=threshold, plot=True)  # Assuming plotDC uses plt.show()
    st.pyplot()

    # Calculate and display ATMV
    st.write('## Absolute Theoretical Movement Value (ATMV)')
    st.write('The ATMV is a measure of the price movement between directional changes relative to the threshold.')
    atmvs = dc.calculateATMV(threshold=threshold)
    atmv_df, _ = convert_to_dataframe(atmvs, [])
    st.dataframe(atmv_df)

    # Calculate and display AR
    st.write('## Adjusted Return (AR)')
    st.write('Adjusted Return is the rate of return for each directional change, normalized by its time duration.')
    ars = dc.calculateAR(threshold=threshold)
    _, ar_df = convert_to_dataframe([], ars)
    st.dataframe(ar_df)

    # Calculate and display Coastline
    st.write('## Coastline')
    st.write('The Coastline is the cumulative sum of all the ATMVs and represents the overall price movement.')
    coastline = dc.calculateCoastline(threshold=threshold)
    st.write(coastline)

    # Calculate and display Number of Directional Changes (NDC)
    st.write('## Number of Directional Changes (NDC)')
    st.write('The NDC is the total count of directional changes and can serve as an indicator of market volatility.')
    ndc = dc.calculateNDC(threshold=threshold)
    st.write(ndc)

if __name__ == '__main__':
    main()



