import pymysql
import sqlalchemy
from sqlalchemy import create_engine

import pandas as pd

# csv > DB
def pushToDB(file, password, dbName, tableName, if_exists):
    df = pd.read_csv(file)
    conn = pymysql.connect(host='localhost', user='root', password=password, db='dbName', charset='utf8')
    cur = conn.cursor()
    dbConnPath = f'mysql+pymysql://root:{password}@localhost/{dbName}'
    dbConn = create_engine(dbConnPath)
    conn = dbConn.connect()
    df.to_sql(name=tableName, con=dbConn, if_exists=if_exists, index=False)


# DB > dataframe
def pullFromDB(password, dbName, tableName):
    conn = pymysql.connect(host='localhost', user='root', password=password, db=dbName, charset='utf8')
    df = pd.read_sql(f'SELECT * FROM {tableName}', con=conn)
    return df