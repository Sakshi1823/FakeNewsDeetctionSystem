from eventregistry import *
from threading import Thread, Lock
from py_ms_cognitive import PyMsCognitiveWebSearch
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import nltk
import pandas as pd
import sys

''' ibm_watson setup 
change Api key in authenticator and service URL '''

authenticator = IAMAuthenticator('aYwDWbWfxdlt8KgpZRj53VKJsofafjh80lVH9kxclTtX')
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url('https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/7e1fa7ba-2be7-478c-8265-ca795824491d')

# event_registry setup
api_key = '492ae2f5-4d0a-445c-8063-a96c5876c1e2'
er = EventRegistry(apiKey = api_key)

global_df = pd.DataFrame()
mutex = Lock()
global_claim = ''


# Given keywords, this funciton appends the article metadata to the global pandas dataframe
def get_articles(keyword):
    global global_df
    global global_claim
    q = QueryArticlesIter(keywords=QueryItems.AND(keyword))
    q.setRequestedResult(RequestArticlesInfo(count = 5, sortBy="sourceImportance"))

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

    mutex.acquire()
    try:
        global_df = pd.concat([global_df,local_df])
    finally:
        mutex.release()

# Given a url, this function returns up to 15 keywords, keyword relevance might pose issue cause if
# relevance is set too high might get empty list lets set for 0.50 right now

def watson (url):
    response = nlu.analyze(
        url=url,
        features=Features(keywords=KeywordsOptions(limit=15))
    ).get_result()
    keywords = []
    for keyword in response['keywords']:
        if keyword['relevance'] > 0.50 and len(keywords) < 8:
            keywords.append(keyword['text'].encode('utf-8'))
    return keywords

# Worker thread class override
class myThread(Thread):
    def __init__(self, query):
        Thread.__init__(self)
        self.query = query

    def run(self):
        get_articles(self.query)

# given claim, azure returns related urls using bing searches
def azure_search(claim):
    search_term = claim
    search_service = PyMsCognitiveWebSearch('75d1a40af4bf4ba4bdf561ae25b5db5c', claim)
    first_three_result = search_service.search(limit=3, format='json') #1-50

    urls = []
   # To get individual result json:
    for i in first_three_result:
        urls.append(i.url.encode('utf-8'))
    return urls

# given a list of urls, this function returns all related keywords for the urls
def azure_claim(urls):
    keywords = []
    for url in urls:
        keywords.append(watson(url))
    return keywords

# given keywords, query event registry and append to global dataframe
def watson_azure_scrape(keywords):
    global global_df

    index = 0
    threads = []

    for query in keywords:
        threads.append(myThread(query))
        threads[index].start()
        index += 1
    for thread in threads:
        thread.join()
    global_df = global_df.reset_index(drop=True)
    print (global_df.shape)
    global_df.to_csv('watson_articles.csv')
#     global_df['uid'] = range(len(global_df.index))
#     return global_df.to_dict(orient='records')

# Call this function with a claim to query event registry
def run_azure(claim):
    claim_tokens = nltk.word_tokenize(claim)
    if len(claim_tokens) == 3:
        # Go straight to event registry with claim
        watson_azure_scrape(claim)
    else:
        watson_azure_scrape(azure_claim(azure_search(claim)))

# Call this function with a url to query event registry
def watson_scrape(url):
    global global_df
    global global_claim
    print ("Getting keywords")
    keywords = watson(url)
    print (keywords)

    index = 0
    threads = []

    for query in keywords:
        print("making threads")
        threads.append(myThread(query))
        threads[index].start()
        print ("getting articles")
        index += 1
    for thread in threads:
        thread.join()
    global_df = global_df.reset_index(drop=True)
    # global_df.to_csv('watson_articles.csv')
    global_df['id'] = range(len(global_df.index))
    bodies = global_df.loc[:,['id','text']]
    bodies.columns = ['BodyID','text']
    bodies.to_csv('bodies.csv')
    claim = [global_claim] * len(global_df.index)
    claims = pd.DataFrame(claim)
    claims['BodyID'] = range(len(global_df.index))
    claims.columns = ['Headlines','BodyID']
    claims.to_csv('claims.csv')
    urls = global_df.loc[:,['id','source','url']]
    urls.to_csv('url.csv')
    print("asdfasdfa")

    print(global_df)
    return global_df.to_dict(orient='records')

def start(type_param, userInput):
    # Your logic to handle the parameters and perform the scraping
    if type_param == 'url':
        watson_scrape(userInput)
        print("asdfasdfaffdsafasdfasdf")
    else:
        run_azure(userInput)
        print("uersyhurtg")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: watson_scraper.py run <type_param> <input_param>")
        sys.exit(1)
    
    # Extract command-line arguments
    type_param = sys.argv[2]
    userInput = sys.argv[3]

    # Call the run function with the extracted parameters
    start(type_param, userInput)

# trash
# def run (argv):
#     print("args 1")
#     print(argv[1])
#     if argv[1] == 'url':
#         print("args 2")
#         print(argv[2])
#         watson_scrape(argv[2])
#         print("asdfasdfaffdsafasdfasdf")
#     else:
#         run_azure(argv[2])


