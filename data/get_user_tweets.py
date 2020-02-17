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


def get_all_tweets(username, info):
    consumer_key = info["consumer_key"]
    consumer_secret = info["consumer_secret"]
    access_key = info["access_key"]
    access_secret = info["access_secret"]

    # Twitter allows access to only 3240 tweets via this method

    # Authorization and initialization

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    number_of_tweets = 3200

    tweets_for_csv = []

    for tweet in tweepy.Cursor(api.user_timeline, screen_name=username, include_rts=False).items(number_of_tweets):
        tweets_for_csv.append([username, tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")])
        print(tweet.text)
        print(tweet.created_at)

    return tweets_for_csv


def write_tweets_csv(filename, tweets):
    try:
        with open(filename, "w") as outfile:
            writer = csv.writer(outfile, delimiter=",")
            writer.writerows(tweets)
    except Exception as e:
        print("Exception Raised-", str(e))


if __name__ == "__main__":
    twitter_cred = dict()
    twitter_cred["consumer_key"] = "PPq7v8RdB2yUaYDH4UgXICfKf"
    twitter_cred["consumer_secret"] = "ZKmr9UjpiaeGBdhITcMbsN7MsQEgWECZUC3fbhQMgPEY1ZNZbF"
    twitter_cred["access_key"] = "2211307145-qbOLwohdFQO2BxA7yjlAG3XrFxtPYltqXdzGWia"
    twitter_cred["access_secret"] = "SuXewv5enPTAeaSOJdOx17tu23DXd03tyrv9W9NEVQQiR"
    json_filename = "twitter_credentials.json"
    save_twitter_cred(json_filename, twitter_cred)
    info = read_twitter_cred(json_filename)
    username = "cambrasine"
    tweets = get_all_tweets(username, info)
    write_tweets_csv(username + "_tweets.csv", tweets)
