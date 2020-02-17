# -*- coding: utf-8 -*-
import csv
import tweepy
import json


def save_twitter_cred(filename, cred):
    try:
        with open(filename, "w") as info:
            json.dump(cred, info, indent=4, sort_keys=True)
            print("saving twitter credentials")
    except Exception as e:
        print("Exception Raised -", str(e))


def read_twitter_cred(filename):
    try:
        with open(filename) as cred_data:
            print("readin twiter credentials")
            info = json.load(cred_data)
    except Exception as e:
        print("Exeption Raised-", str(e))
    return info


def get_all_tweets(screen_name, info):
    consumer_key = info["consumer_key"]
    consumer_secret = info["consumer_secret"]
    access_key = info["access_key"]
    access_secret = info["access_secret"]

    # Twitter allows access to only 3240 tweets via this method

    # Authorization and initialization

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialization of a list to hold all Tweets

    all_the_tweets = []

    # We will get the tweets with multiple requests of 200 tweets each
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    for tweet in new_tweets:
        print(tweet.text)
    # saving the most recent tweets

    all_the_tweets.extend(new_tweets)

    # save id of 1 less than the oldest tweet

    oldest_tweet = all_the_tweets[-1].id - 1

    # grabbing tweets till none are left

    new_tweets = api.user_timeline(screen_name=screen_name, max_id=oldest_tweet)
    """ 
    while len(new_tweets) < 1000:
        # The max_id param will be used subsequently to prevent duplicates

        new_tweets = api.user_timeline(screen_name=screen_name, max_id=oldest_tweet)
        print(new_tweets)
    """
    # save most recent tweets

    all_the_tweets.extend(new_tweets)

    # id is updated to oldest tweet - 1 to keep track

    oldest_tweet = all_the_tweets[-1].id - 1
    print('...%s tweets have been downloaded so far' % len(all_the_tweets))

    # transforming the tweets into a 2D array that will be used to populate the csv

    outtweets = [[tweet.id_str, tweet.created_at,
                  tweet.text.encode('utf-8')] for tweet in all_the_tweets]

    # writing to the csv file

    with open(screen_name + '_tweets.csv', 'w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'created_at', 'text'])
        writer.writerows(outtweets)


if __name__ == "__main__":
    twitter_cred = dict()
    twitter_cred["consumer_key"] = "PPq7v8RdB2yUaYDH4UgXICfKf"
    twitter_cred["consumer_secret"] = "ZKmr9UjpiaeGBdhITcMbsN7MsQEgWECZUC3fbhQMgPEY1ZNZbF"
    twitter_cred["access_key"] = "2211307145-qbOLwohdFQO2BxA7yjlAG3XrFxtPYltqXdzGWia"
    twitter_cred["access_secret"] = "SuXewv5enPTAeaSOJdOx17tu23DXd03tyrv9W9NEVQQiR"
    json_filename = "twitter_credentials.json"
    save_twitter_cred(json_filename, twitter_cred)
    info = read_twitter_cred(json_filename)
    get_all_tweets("cambrasine", info)
