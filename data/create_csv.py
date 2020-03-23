import os
import csv
import json
from datetime import datetime

if __name__ == "__main__":
    orig_path = "trump_tweets/"
    orig_files = os.listdir(path=orig_path)
    orig_data = []
    for file in orig_files:
        try:
            with open(orig_path + file) as tweet:
                orig_data.append(json.load(tweet))
        except Exception as e:
            print("Exception Raised-", str(e))
    header = ["ID", "usernameTweet",  "datetime", "text", "url", "nbr_retweet", "nbr_favorite", "nbr_reply", "is_reply", "is_retweet", "user_id"]
    csv_rows = []
    for item in orig_data:
        row = []
        print(item)
        for name in header:
            row.append(item[name])
        if item["usernameTweet"] == "realDonaldTrump":
            csv_rows.append(row)

    csv_rows.sort(key=lambda date: datetime.strptime(date[2], "%Y-%m-%d %H:%M:%S"))

    print(csv_rows)
    with open("trump_tweets.csv", "w") as csv_out:
        writer = csv.writer(csv_out, delimiter=",")
        writer.writerow(header)
        writer.writerows(csv_rows)
