import wrds
import pandas as pd

conn = wrds.Connection()

def compCusip():
    """
    Function used to find the CUSIP9 for a specific exchange Q is for NASDAQ. It returns the query of the CUSIP9
    """
    query = f"""
            select cusip9
            from crsp.stocknames_v2
            where primaryexch='Q' and cusip9 is not null
            """
    c=conn.raw_sql(query)
    return c

def compInfo(comp_cusips, date):
    """
    Function used to take as arguments the CUSIP9 as a list and the date used to find the query. The date must be in the form: current_year-1

    It return the items preselected in the query.
    """
    query = f"""
            SELECT ITEM5601, ITEM7011, ITEM7210, ITEM7220, ITEM7230, ITEM7240, ITEM7250, ITEM8101, ITEM8106, ITEM8111, ITEM8121, ITEM8136, ITEM8226, ITEM8231, ITEM8236, ITEM8306, ITEM8316, ITEM8336, ITEM8366, ITEM8371, ITEM8401, ITEM8406, ITEM8601, ITEM8611, ITEM8621, ITEM8626, ITEM8631, ITEM8636, ITEM6004
            FROM trws.wrds_ws_funda
            WHERE item6004 IN {tuple(comp_cusips)} AND ITEM5601 IS NOT NULL
            and year_>={date}
            """
    c = conn.raw_sql(query)
    c.fillna(method='ffill', inplace=True)
    c.fillna(method="bfill", inplace=True)
    return c
    
def getCompInfo(year):
    """
    Function to give the data specified for the NASDAQ
    """
    # convert the cusip9 obtained to a list in order to pass it easily in a query
    comp_cusips = compCusip()["cusip9"].tolist()
    #if the dataset is empy, there will be an error
    if comp_cusips:
        df=[compInfo(comp_cusips,year)]
        r_df=pd.concat(df, ignore_index=True).drop_duplicates(subset='item5601', keep='first')
        #choose between print or a csv output
        print(r_df)
        #r_df.to_csv("output.csv")
    else: print("No data found in the current request")

getCompInfo(2022)
conn.close()