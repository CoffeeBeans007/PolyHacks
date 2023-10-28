import wrds
import pandas as pd

class NasdaqData:
    def __init__(self):
        """Initialize the connection to the WRDS database."""
        self.conn = wrds.Connection()
        
    def compCusip(self):
        """Return the CUSIP9 for a specific exchange Q (NASDAQ)."""
        query = """
                SELECT cusip9
                FROM crsp.stocknames_v2
                WHERE primaryexch='Q' AND cusip9 IS NOT NULL
                """
        return self.conn.raw_sql(query)

    def compInfo(self, comp_cusips, date):
        """Return the items preselected in the query for given CUSIP9 and date."""
        query = f"""
                SELECT ITEM5601, ITEM7011, ITEM7210, ITEM7220, ITEM7230, ITEM7240, 
                       ITEM7250, ITEM8101, ITEM8106, ITEM8111, ITEM8121, ITEM8136, 
                       ITEM8226, ITEM8231, ITEM8236, ITEM8306, ITEM8316, ITEM8336, 
                       ITEM8366, ITEM8371, ITEM8401, ITEM8406, ITEM8601, ITEM8611, 
                       ITEM8621, ITEM8626, ITEM8631, ITEM8636, ITEM6004
                FROM trws.wrds_ws_funda
                WHERE item6004 IN {tuple(comp_cusips)} AND ITEM5601 IS NOT NULL
                AND year_>={date}
                """
        c = self.conn.raw_sql(query)
        c.fillna(method='ffill', inplace=True)
        c.fillna(method="bfill", inplace=True)
        return c

    def getCompInfo(self, year):
        """Return the data specified for the NASDAQ for a given year."""
        comp_cusips = self.compCusip()["cusip9"].tolist()
        if comp_cusips:
            df = [self.compInfo(comp_cusips, year)]
            r_df = pd.concat(df, ignore_index=True).drop_duplicates(subset='item5601', keep='first')
            self.data = r_df
            print(r_df)
            # r_df.to_csv("output.csv")
            
        else: 
            print("No data found in the current request")
            return None

    def close(self):
        """Close the connection to the WRDS database."""
        self.conn.close()





