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
            #print (df.info())
            frames.append(df)
            '''
            for i,r in df.iterrows():
                _keys = []
                _form = {}
                for k in r.keys():
                    _form[str(k)] = r[k]
                _forms.append(_form)
            '''

    #print (len(_forms))
    #print (frames)
    result = pd.concat(frames)
    print ('result df')
    print (result.info())
    result.to_csv ('output.csv', index = False, header=True)



def filterdata():
    df = pd.read_csv('output.csv', dtype='unicode')

    print (df.info())
    print (df.head())
    #print (df.iloc[0])
    #print (df.iloc[0]['Final Outcome Combined'])

    df1 = df[df['Final Outcome\nCombined'].notnull()]
    print (df.info())
    #df1 = df[df['Final Outcome Combined'].isnull()]

    df1.to_csv ('filtered_1.csv', index = False, header=True)

    #df2 = df[df['Final Outcome Combined'].notnull()]
    df2 = df[df['Final Outcome\nCombined'].isnull()]
    print (df.info())

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
    df2 = pd.read_csv('_ex.csv', dtype='unicode')
    frames.append(df2)
    result = pd.concat(frames)
    print ('new result df')
    print (result.info())
    result.to_csv ('output2.csv', index = False, header=True)

def preprocessed():
    
    #df = pd.read_csv('filtered_1.csv', dtype='unicode')
    df = pd.read_csv('output2.csv', dtype='unicode') # this is the final file!
    print (df.info())
    print (df.head())
    print (df.iloc[0])

#init()
#filterdata()
#load()
combine()
#preprocessed()
