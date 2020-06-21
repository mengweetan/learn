import pandas as pd
import numpy as np


def init():
    __filename = 'raw1.xlsx'
    xls_file = pd.ExcelFile(__filename)
    dfs  = {sheet_name: xls_file.parse(sheet_name)
              for sheet_name in xls_file.sheet_names}

    _forms = []
    frames = []
    for sheet_key,v in dfs.items():
        if 'master' in sheet_key.lower():
            print (sheet_key)
            df = dfs[sheet_key]
            df.to_csv('backup.csv')
            df = pd.read_csv('backup.csv', skiprows=[0], dtype='unicode')

            frames.append(df)
 


    result = pd.concat(frames)

    result.to_csv ('output.csv', index = False, header=True)



def filterdata():
    df = pd.read_csv('output.csv', dtype='unicode')


    df1 = df[df['Final Outcome\nCombined'].notnull()]
    df1.to_csv ('filtered_1.csv', index = False, header=True)

    df2 = df[df['Final Outcome\nCombined'].isnull()]
    df2.to_csv ('filtered_1_null.csv', index = False, header=True)


def load():
    df = pd.read_csv('filtered.csv', dtype='unicode')

    print (df.info())
    print (df.head())
    print (df.iloc[0])

    df = pd.read_csv('filtered_null.csv', dtype='unicode')

    print (df.info())
    print (df.head())
    print (df.iloc[0])

def combine():
    frames = []
    df = pd.read_csv('filtered.csv', dtype='unicode')
    frames.append(df)
    df2 = pd.read_csv('_ex.csv', dtype='unicode')  # job done by hand...
    frames.append(df2)
    result = pd.concat(frames)

    result.to_csv ('output2.csv', index = False, header=True)



#init()
#filterdata()
#load()
combine()

