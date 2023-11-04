from typing import List, Union
import pandas as pd

class RebalanceScheduler:
    def __init__(self, end_date: str, first_rebalance_year: int, rebalance_month: int,
                 rebalance_week: int, rebalance_weekday: int, rebalance_frequency: str):
        self.end_date = pd.Timestamp(end_date)
        self.first_rebalance_year = first_rebalance_year
        self.rebalance_month = rebalance_month
        self.rebalance_week = rebalance_week
        self.rebalance_weekday = self.map_weekday(weekday=rebalance_weekday)  # Monday is 0, Sunday is 6
        self.rebalance_frequency = rebalance_frequency  # 'M' (Monthly), 'Q' (Quarterly), 'S' (Semi-Annual), 'A' (Annual)
        self.initial_rebalance_date = self.generate_initial_rebalance_date()
        self.rebalance_dates = self.generate_rebalance_dates()
        self.next_rebalance = None
        self.previous_rebalance = None
        self.set_next_and_previous_rebalance()

    def map_weekday(self, weekday: Union[int, str]) -> int:
        """
        Maps a weekday represented as a string to its corresponding integer index.
        """
        if isinstance(weekday, str):
            weekday_str_to_int = {
                'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
                'FRI': 4, 'SAT': 5, 'SUN': 6
            }
            return weekday_str_to_int[weekday.upper()]
        return weekday

    def generate_initial_rebalance_date(self) -> pd.Timestamp:
        """
        Generates the initial rebalance date based on the year, month, week, and weekday specified.
        """
        first_day_of_month = pd.Timestamp(year=self.first_rebalance_year, month=self.rebalance_month, day=1)
        first_weekday = first_day_of_month.weekday()
        # Calculate which day of the month the rebalance weekday falls in the specified week
        first_occurrence = self.rebalance_weekday - first_weekday
        first_occurrence = first_occurrence + 7 if first_occurrence < 0 else first_occurrence
        day_of_month = first_occurrence + 1 + (self.rebalance_week - 1) * 7
        initial_rebalance_date = pd.Timestamp(year=self.first_rebalance_year, month=self.rebalance_month, day=day_of_month)
        return initial_rebalance_date

    def generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Generates rebalance dates aligned with the initial_rebalance_date date,
        considering the specified frequency until the end_date.
        """
        freq_mapping = {'M': 1, 'Q': 3, 'S': 6, 'A': 12}
        rebalance_dates = [self.initial_rebalance_date]

        # Create subsequent rebalance dates by adding frequency in months and aligning to the same week and weekday
        next_date = self.initial_rebalance_date
        while next_date <= self.end_date:
            months_to_add = freq_mapping[self.rebalance_frequency]
            next_date = pd.Timestamp(next_date.year, next_date.month, next_date.day) + pd.DateOffset(
                months=months_to_add)
            # Re-align day to the same week and weekday of the month as the initial rebalance date
            first_day_of_next_month = pd.Timestamp(year=next_date.year, month=next_date.month, day=1)
            first_weekday = first_day_of_next_month.weekday()
            first_occurrence = self.rebalance_weekday - first_weekday
            first_occurrence = first_occurrence + 7 if first_occurrence < 0 else first_occurrence
            day_of_month = first_occurrence + 1 + (self.rebalance_week - 1) * 7

            # Handle invalid dates by reducing the day until a valid date is found
            while True:
                try:
                    next_date = pd.Timestamp(year=next_date.year, month=next_date.month, day=day_of_month)
                    break
                except ValueError:
                    day_of_month -= 1

            if next_date <= self.end_date:
                rebalance_dates.append(next_date)

        return rebalance_dates

    def set_next_and_previous_rebalance(self):
        """
        Sets the next and previous rebalance dates dynamically based on current date.
        """
        now = pd.Timestamp.now()
        future_dates = [date for date in self.rebalance_dates if date > now]
        past_dates = [date for date in self.rebalance_dates if date <= now]

        self.next_rebalance = future_dates[0] if future_dates else None
        self.previous_rebalance = past_dates[-1] if past_dates else None

    def get_next_rebalance(self) -> Union[pd.Timestamp, None]:
        """
        Returns the next rebalance date.
        """
        return self.next_rebalance

    def get_previous_rebalance(self) -> Union[pd.Timestamp, None]:
        """
        Returns the previous rebalance date.
        """
        return self.previous_rebalance

    def __repr__(self) -> str:
        """
        Provides a string representation of the object.
        """
        return (f"RebalanceScheduler(end_date={self.end_date}, initial_rebalance_date={self.initial_rebalance_date}, "
                f"rebalance_frequency={self.rebalance_frequency}, next_rebalance={self.next_rebalance}, "
                f"previous_rebalance={self.previous_rebalance})")

if __name__ == "__main__":

    end_date = '2021-12-31'
    first_rebalance_year = 2021
    rebalance_month = 1
    rebalance_week = 1
    rebalance_weekday = 'MON'
    rebalance_frequency = 'M'


    # Test the class
    scheduler = RebalanceScheduler(end_date=end_date, first_rebalance_year=first_rebalance_year,
                                   rebalance_month=rebalance_month, rebalance_week=rebalance_week,
                                   rebalance_weekday=rebalance_weekday, rebalance_frequency=rebalance_frequency)

    print("Generated Rebalance Dates:", scheduler.rebalance_dates)
    print("Next Rebalance:", scheduler.get_next_rebalance())
    print("Previous Rebalance:", scheduler.get_previous_rebalance())
    print("String representation:", scheduler)