import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('-d', '--dataset', type=str, help='The dataset name', required=False)
    args = parser.parse_args()
    
    file = os.getcwd() + "/" + args.dataset + "/result.log"
    lines = None
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines[::-1] if line.strip()]
    re, l2 = lines[0], lines[1]
    print("------------------------------------------------------")
    print("DataSet is {}".format(args.dataset))
    print("RE = {}".format(re))
    print("L2 = {}".format(l2))
