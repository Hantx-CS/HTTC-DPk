import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('-e', '--epsilon', type=str, help='The parivacy budget epsilon', required=False)
    args = parser.parse_args()

    re, l2 = [], []
    for n in range(10000, 100001, 10000):
        file = os.getcwd() + "/IMDB-single-eps" + args.epsilon + "/error_n" + str(n) + ".log"
        with open(file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines[::-1] if line.strip()]
            re.append(float(lines[0]))
            l2.append(float(lines[1]))

    print("------------------------------------------------------")
    print("Epsilon is {}".format(args.epsilon))
    print("RE = {}".format(re))
    print("L2 = {}".format(l2))
