from lib_mongo_retriever_master.mongo import MongoDB
import pandas as pd
from datetime import datetime
from datetime import timedelta

def _charac_recovered(df):
    charac = df.pop('charac')
    return charac

def _ranking(s):
    count = s.value_counts()
    data = {'Charac': count.index, 'Occurencies': count.values}
    output = pd.DataFrame(data=data)
    return output

def main():
    today = datetime.today()
    cursor = MongoDB(database='driftTrendDetection', collection='detection').collection.find({"date": {"$gt": today - timedelta(days=7)}})
    df = pd.DataFrame(list(cursor))
    output = _charac_recovered(df)
    output_ordered = _ranking(output)
    output_ordered.to_csv('Reporting_drift.csv', sep=',')

if __name__ == '__main__':
    main()