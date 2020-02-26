import csv

w_list = []
text = ""
filename = "cambrasine_only_tweets.csv"
with open(filename, "r") as csvin:
    reader = csv.reader(csvin, delimiter=",")

    for row in reader:
        line = row[3]

        if line.endswith(".") or line.endswith("!") or line.endswith("?") or line.endswith("..."):
            text += line
        else:
            text += line + "."

        words = line.split(" ")
        for word in words:
            w_list.append(word)

with open("cambrasine_tweets.txt", "w") as fileout:
    fileout.write(text)

uniq_words = list(set(w_list[1:]))

#print(w_list)
#print(uniq_words)
#print(len(w_list))
#print(len(uniq_words))

with open("cambrasine_vocabulary.txt", "w") as fout:
    for word in uniq_words:
        fout.write(word + "\n")

