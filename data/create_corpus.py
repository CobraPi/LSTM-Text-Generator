import csv

w_list = []
text = ""
filename = "cambrasine_only_tweets.csv"
with open(filename, "r") as csvin:
    reader = csv.reader(csvin, delimiter=",")
    count = 0
    avg = 0
    for row in reader:
        line = row[3]

        if line.endswith(".") or line.endswith("!") or line.endswith("?") or line.endswith("..."):
            text += line + " "
        else:
            text += line + ". "

        words = line.split(" ")
        count += 1
        avg += len(words)
        for word in words:
            w_list.append(word)
    print(avg / count)
    print(len(w_list))

with open("cambrasine_tweets.txt", "w") as fileout:
    fileout.write(text)

uniq_words = list(set(w_list))

with open("cambrasine_vocabulary.txt", "w") as fout:
    for word in uniq_words:
        fout.write(word + "\n")
