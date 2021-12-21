from IPython import embed
import random

if __name__ == "__main__":
    adjectives = []
    f = open("adjectives.txt", "r")
    for x in f:
        if ("\n" in x):
            x = x.split("\n")[0]
        adjectives.append(x) 

    colors = []
    f = open("colors.txt", "r")
    for x in f:
        if ("\n" in x):
            x = x.split("\n")[0]
        colors.append(x) 

    nouns = []
    f = open("nouns.txt", "r")
    for x in f:
        if ("\n" in x):
            x = x.split("\n")[0]
        nouns.append(x) 

    # enumerate all the possibilities
    uid = []
    for adj in adjectives:
        for col in colors:
            for n in nouns:
                uuid = (adj+col+n+"\n").lower()
                uid.append(uuid)

    random.shuffle(uid)

    with open("./uuids.txt", "w+") as fp:
        [fp.write(entry) for entry in uid]

    with open("./uuid_count.txt", "w+") as fp:
        fp.write("0")

    
    