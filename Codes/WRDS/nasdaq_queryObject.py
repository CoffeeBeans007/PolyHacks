# Import necessary libraries
import wrds
import pandas as pd

# Define a class named NasdaqData
class NasdaqData:
    def __init__(self):
        """Initialize the connection to the WRDS database."""
        self.conn = wrds.Connection()  # Establish a connection to the WRDS database upon class instantiation

    def compCusip(self, index='Q'):
        """
        Return the CUSIP9 for a specific exchange.
        
        list of possible values: 
        A	NYSE American
        B	BATS
        C	CONSOLIDATED
        I	IEX
        N	NYSE
        N/A	Not Applicable
        Q	NASDAQ
        R	NYSE ARCA
        X	Unknown
        """
        # SQL query to retrieve CUSIP9 codes for a specific exchange from the WRDS database
        query = f"""
                SELECT cusip9
                FROM crsp.stocknames_v2
                WHERE primaryexch='{index}' AND cusip9 IS NOT NULL
                """
        return self.conn.raw_sql(query)  # Execute the SQL query and return the results

    def compInfo(self, comp_cusips, date, nb_companies=100):
        """Return the items preselected in the query for given CUSIP9 and date."""
        # SQL query to retrieve specific company information based on CUSIP9 codes and date from the WRDS database
        query = f"""
                SELECT ITEM7011, ITEM7220, ITEM7230, ITEM7240, ITEM7210,
                ITEM7250, ITEM8101, ITEM8106, ITEM8111, ITEM8121, ITEM8136, 
                ITEM8226, ITEM8231, ITEM8236, ITEM8306, ITEM8316, ITEM8336, 
                ITEM8366, ITEM8371, ITEM8401, ITEM8406, ITEM8601, ITEM8611, 
                ITEM8621, ITEM8626, ITEM8631, ITEM8636, item6004, ITEM5601
                FROM trws.wrds_ws_funda
                WHERE ITEM6004 IN {tuple(comp_cusips)} AND ITEM5601 IS NOT NULL 
                    AND year_ >= {date}
                ORDER BY year_ ASC
                LIMIT {nb_companies}
                """
        c = self.conn.raw_sql(query)  # Execute the SQL query and store the results in variable c
        c.fillna(method='ffill', inplace=True)  # Forward fill missing values in the DataFrame
        c.fillna(method="bfill", inplace=True)  # Backward fill remaining missing values
        return c  # Return the cleaned DataFrame containing company information

    def getCompInfo(self, year, nb_comp):
        """Return the data specified for the NASDAQ for a given year."""
        comp_cusips = self.compCusip()["cusip9"].tolist()  # Retrieve CUSIP9 codes for NASDAQ-listed companies
        if comp_cusips:  # If CUSIP9 codes are found
            df = [self.compInfo(comp_cusips, year, nb_comp)]  # Retrieve company information for the specified year and number of companies
            r_df = pd.concat(df, ignore_index=True)  # Concatenate the data into a single DataFrame
            self.data = r_df  # Store the concatenated DataFrame in the class attribute 'data'
            print(r_df)  # Print the concatenated DataFrame
        else: 
            print("No data found in the current request")  # If no CUSIP9 codes are found, print a message
            return None  # Return None

    def close(self):
        """Close the connection to the WRDS database."""
        self.conn.close()  # Close the connection to the WRDS database when called

# Main block of code
if __name__ == '__main__':
    comp = NasdaqData()  # Create an instance of the NasdaqData class
    data = comp.getCompInfo(2008, 1000)  # Call the getCompInfo method to retrieve company information for the year 2008 and 1000 companies
    if data is not None:  # If data is retrieved successfully
        print(data)  # Print the retrieved data
