import os
import csv
import json

if __name__ == "__main__":
    files = os.listdir(path="Test")

    data = []
    print(files)
    for file in files:
        try:
            with open("Test/" + file) as tweet:
                data.append(json.load(tweet))
        except Exception as e:
            print("Exception Raised-", str(e))

    header = ["usernameTweet", "ID", "text", "url", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "is_reply", "is_retweet", "user_id"]
    csv_rows = []
    for item in data:
        row = []
        for name in header:
            row.append(item[name])
        csv_rows.append(row)

    with open("cambrasine_tweets.csv", "w") as csv_out:
        writer = csv.writer(csv_out, delimiter=",")
        writer.writerow(header)
        writer.writerows(csv_rows)
