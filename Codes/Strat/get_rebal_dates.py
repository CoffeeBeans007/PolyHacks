import pandas as pd
from typing import List, Union


class GetRebalDates:
    def __init__(self, termination_date: str, initial_year: int, reb_month: int,
                 reb_week: int, reb_weekday: int, reb_frequency: str):
        self.termination_date = pd.Timestamp(termination_date)
        self.initial_year = initial_year
        self.reb_month = reb_month
        self.reb_week = reb_week
        self.reb_weekday = self._weekday_conversion(weekday=reb_weekday)
        self.reb_frequency = reb_frequency
        self.start_reb_date = self._determine_start_date()
        self.reb_dates = self._generate_reb_dates()

    def _weekday_conversion(self, weekday: Union[int, str]) -> int:
        if isinstance(weekday, str):
            return {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}[weekday.upper()]
        return weekday

    def _determine_start_date(self) -> pd.Timestamp:
        start_of_month = pd.Timestamp(year=self.initial_year, month=self.reb_month, day=1)
        first_occurrence = self.reb_weekday - start_of_month.weekday() + 1 + (self.reb_week - 1) * 7
        while True:
            try:
                return pd.Timestamp(year=self.initial_year, month=self.reb_month, day=first_occurrence)
            except ValueError:
                first_occurrence -= 1

    def _generate_reb_dates(self) -> List[pd.Timestamp]:
        frequency_dict = {'M': 1, 'Q': 3, 'S': 6, 'A': 12}
        reb_dates = [self.start_reb_date]
        subsequent_date = self.start_reb_date

        while True:
            months_to_add = frequency_dict[self.reb_frequency]
            subsequent_date = subsequent_date + pd.DateOffset(months=months_to_add)

            # Re-align to the same week and weekday
            first_day_of_next_month = pd.Timestamp(year=subsequent_date.year, month=subsequent_date.month, day=1)
            first_occurrence = self.reb_weekday - first_day_of_next_month.weekday()
            first_occurrence = first_occurrence + 7 if first_occurrence < 0 else first_occurrence
            day_of_month = first_occurrence + 1 + (self.reb_week - 1) * 7

            while True:
                try:
                    subsequent_date = pd.Timestamp(year=subsequent_date.year, month=subsequent_date.month,
                                                   day=day_of_month)
                    break
                except ValueError:
                    day_of_month -= 1

            if subsequent_date > self.termination_date:
                break

            reb_dates.append(subsequent_date)

        return reb_dates


if __name__ == "__main__":
    # Création d'une instance de GetRebalDates
    scheduler = GetRebalDates(
        termination_date='2021-12-31',
        initial_year=2010,
        reb_month=3,
        reb_week=1,
        reb_weekday='MON',
        reb_frequency='Q'
    )
    # Récupération et affichage des dates de rebalancement
    rebalance_dates = scheduler.reb_dates

    print("Dates de rebalancement :")
    for date in rebalance_dates:
        print(date)

