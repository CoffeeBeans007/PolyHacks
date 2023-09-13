import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                        # if last state is upward DC event, current sate should be changed into upward OS event.
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

    def plotDC(self, threshold, plot=True):
        '''
        Visualizing DC upon TS data.
        '''
        df = self.dc_pattern(threshold)

        df['ts'] = list(self.data['Close'])
        df['shift'] = df.iloc[:, 1].shift(-1)
        begin = df.iloc[0, :]
        listPlot = [df.index[0][0], begin['ts']]
        if plot: plt.figure(figsize=(25, 12))
        self.listExtreme = [listPlot]
        x_all = df['ts']
        if plot: plt.plot(x_all.index.map(lambda x: x[0]), x_all, 'cornflowerblue')
        for idx, row in df.iterrows():

            if idx == 0:
                continue
            time_str = idx[0].to_pydatetime().date()
            if row[1] == 'up_OS' and row[-1] == 'down_DC':
                self.listExtreme.append([idx[0], row['ts']])
                if plot:
                    plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'lightgreen')
                listPlot = [idx[0], row['ts']]

            if row[1] == 'down_DC' and row[-1] == 'down_OS':
                if plot:
                    plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'tomato')
                listPlot = [idx[0], row['ts']]

            if row[1] == 'down_OS' and row[-1] == 'up_DC':
                self.listExtreme.append([idx[0], row['ts']])
                if plot:
                    plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'lightgreen')
                listPlot = [idx[0], row['ts']]

            if row[1] == 'up_DC' and row[-1] == 'up_OS':
                if plot:
                    plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'tomato')
                listPlot = [idx[0], row['ts']]

            if row[1] == 'up_DC' and row[-1] == 'down_DC':
                self.listExtreme.append([idx[0], row['ts']])
                if plot: plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'tomato')
                listPlot = [idx[0], row['ts']]

            if row[1] == 'down_DC' and row[-1] == 'up_DC':
                self.listExtreme.append([idx[0], row['ts']])
                if plot: plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'tomato')
                listPlot = [idx[0], row['ts']]

        if plot: plt.plot([listPlot[0], idx[0]], [listPlot[1], row['ts']], 'lightgreen')
        self.listExtreme.append([idx[0], row['ts']])

        if plot: plt.show()

    def calculateATMV(self, threshold):
        aTMV = []
        self.plotDC(threshold, plot=False)
        previous = self.listExtreme[0]
        for turningPoint in self.listExtreme[1:]:
            currentATMV = (turningPoint[1] - previous[1]) / previous[1] / threshold

            aTMV.append([[previous[0], turningPoint[0]], np.abs(currentATMV)])
            previous = turningPoint
        return aTMV

    def calculateAR(self, threshold):
        aTMV = self.calculateATMV(threshold)
        aR = list(map(lambda x: x[1] / ((x[0][1] - x[0][0]).days), aTMV))
        timeSpan = list(zip(*aTMV))[0]
        aR = list(zip(timeSpan, aR))
        return aR

    def calculateCoastline(self, threshold):
        aTMV = self.calculateATMV(threshold)
        Coastline = sum(list(zip(*aTMV))[1])

        return Coastline

    def calculateNDC(self, threshold):
        return len(self.calculateATMV(threshold))