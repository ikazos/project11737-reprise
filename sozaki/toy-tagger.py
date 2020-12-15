import sys

path = sys.argv[1]
token = sys.argv[2]

with open(path, "r") as f:
    for line in f:
        print("LANG_{} {}".format(token.upper(), line), end="")