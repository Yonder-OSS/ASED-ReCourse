import re
import os
import copy
import random

import sys

from elasticsearch import Elasticsearch
from datetime import datetime, date, timedelta
import time
from creds import *
import pandas as pd

# Helper function for generating a client for connection to ES
def get_elastic_client(username, password):
    return Elasticsearch([{'host': 'elastic-analyst.ops.newknowledge.io', 'port': 9200}],http_auth=(username, password))

def get_data(es, query, index='discover_*'):
    results = es.search(index=index, body=query, scroll='5m', request_timeout=30)

    sc_id = results['_scroll_id']
    all_tweets = []
    num_res = 1
    start = datetime.now()
    try:
        i = 1
        while num_res > 0:
            print('Fetching batch', i)
            res = es.scroll(scroll_id=sc_id, scroll='2m', request_timeout=30)
            sc_id = res['_scroll_id']
            num_res = len(res['hits']['hits'])
            all_tweets.extend([r['_source'] for r in res['hits']['hits']])
            i += 1
    finally:
        es.clear_scroll(scroll_id=sc_id)

    return [(tweet['authorScreenName'], tweet['content'], tweet['createdAt'], tweet['hashtags'], tweet['domains'],
             tweet['urls'], tweet['parentScreenName'], tweet['authorUserId'], tweet['parentContentId'],
             tweet['parentUserId'], tweet['platform']) \
            for tweet in all_tweets if tweet['content'] and not tweet['connectionType'] == 'retweet']

def get_data_for_day_elastic(es=None, platform='twitter_stream', index='discover_*', project='*', start_day=0, end_day=0):
    if es == None:
        raise Exception("Elastic connection object not provided")

    query = {
        "query": {
            "bool":{
                "must":{
                    "term": {"project": project}
                },
                "filter": {
                    "range":{
                        "createdAt": {
                            "gte": 'now-{0}d/d'.format(start_day),
                            "lt": 'now-{0}d/d'.format(end_day)
                        }
                    }
                }
                
            }
        },
        "size": 10000
    }
    return get_data(es, query, index=index)

# main portion of script
print(sys.version)

es = get_elastic_client('garrett','GYhc5xFvEpDcbG5x$n+yd')

start_time = time.time()
e_tweets = get_data_for_day_elastic(es, index='discover_business', project='hamiltonredux', start_day=2, end_day=0)
print(time.time() - start_time)

e = pd.DataFrame(e_tweets)
e.columns = ['user_screen_name', 'text', 'created_at', 'hashtags', 'domains', 'urls', 'rt_screen_name', 'user_id_str', 'rt_id_str', 'rt_user_id_str', 'platform']

e.head(10)

e.to_csv('tweets_2days.csv', index=False)