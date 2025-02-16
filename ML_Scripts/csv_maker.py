from eventregistry import *
import pandas as pd

# event_registry setup
api_key = '492ae2f5-4d0a-445c-8063-a96c5876c1e2'
er = EventRegistry(apiKey = api_key)

def get_articles(keywords):
    global global_claim
    q = QueryArticlesIter(keywords=QueryItems.AND(keywords))
    q.setRequestedResult(RequestArticlesInfo(count= 100, sortBy="sourceImportance"))

    x = 0

    local_df = pd.DataFrame()

    res = er.execQuery(q)
    for article in res['articles']['results']:
        if x == 0:
            global_claim = article['title'].encode('utf-8')
        data = {
            'source': article['source']['title'].encode('utf-8'),
            'url' : article['url'].encode('utf-8'),
            'text' : article['body'].encode('utf-8')
        }
        local_df = pd.concat([local_df, pd.DataFrame(data,index=[x])])
        x += 1

    # local_df = local_df.reset_index(drop=True)
    # local_df['id'] = range(len(local_df.index))
    # bodies = local_df.loc[:,['id','text']]
    # bodies.columns = ['BodyID','text']
    # bodies.to_csv('bodies.csv')
    # claim = [global_claim] * len(local_df.index)
    # claims = pd.DataFrame(claim)
    # claims['BodyID'] = range(len(local_df.index))
    # claims.columns = ['Headlines','BodyID']
    # claims.to_csv('claims.csv')
    # urls = local_df.loc[:,['id','source','url']]
    # urls.to_csv('url.csv')
    # print("completed")

    local_df = local_df.reset_index(drop=True)
    local_df['id'] = range(len(local_df.index))
    bodies = local_df.loc[:,['id','text']]
    bodies.columns = ['BodyID','text']
    bodies.to_csv(r'CSVs\bodies.csv',mode='a',header=False,index=True)
    claim = [global_claim] * len(local_df.index)
    claims = pd.DataFrame(claim)
    claims['BodyID'] = range(len(local_df.index))
    claims.columns = ['Headlines','BodyID']
    claims.to_csv(r'CSVs\claims.csv',mode='a',header=False,index=True)
    urls = local_df.loc[:,['id','source','url']]
    urls.to_csv(r'CSVs\url.csv',mode='a',header=False,index=True)
    print("completed")

'''

[b'Congress leader Rahul Gandhi', b'Prime Minister Narendra Modi', b'caste census', b'Congress leader',
 b'per cent tribals', b'last year\xe2\x80\x99s Bharat Jodo Yatra',
   b'Union minister Pralhad Joshi', b'Rahul Gandhi']

'''

if __name__ == '__main__':
    print ('start')
    get_articles(keywords='Congress leader Rahul Gandhi')