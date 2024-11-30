#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <tuple>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mt19937ar.h"
#include "MemoryOperation.h"
#include "include/stats.hpp"

using namespace std;

string EdgeFile;
string chosenEdgeFile;
int NodeNum;
double Eps;
string Eps_s;
double EpsNsMaxDeg;
int NSType;
int ItrNum;
int Alg;
double Balloc[3];
char *Balloc_s[3];

// Initialization of statslib
stats::rand_engine_t engine(1776);

FILE *FileOpen(string filename, const char *mode) {
	FILE *fp;

	if ((fp = fopen(filename.c_str(), mode)) == NULL) {
		cout << "cannot open " << filename << endl;
		exit(-1);
	}
	return fp;
}

bool checkFileExistence(const std::string& str) {
    std::ifstream ifs(str);
    return ifs.is_open();
}

// Randomly generate 0, 1, 2, ..., size-1, and store the first num values into rndperm
void MakeRndPerm(int *rndperm, int size, int num) {
	int rnd;
	int *ordperm;
	int i, j;

	// 0, 1, 2, ..., size-1 --> ordperm
	ordperm = (int *)malloc(size * sizeof(int));
	for (i = 0; i < size; i++) {
		ordperm[i] = i;
	}

	for (i = 0; i < num; i++) {
		rnd = genrand_int32() % (size - i);
		rndperm[i] = ordperm[rnd];
		for (j = rnd + 1; j < size - i; j++) {
			ordperm[j - 1] = ordperm[j];
		}
	}

	free(ordperm);
}

// Read edges from the edge file
void ReadEdges(int *node_order){
	int node1, node2;
	int i, j, k;
	char s[1025];
	char *tok;
	FILE *fp;
    vector<pair<int, int>> edges;

    fp = FileOpen(EdgeFile, "r");
	for(i=0;i<3;i++) fgets(s, 1024, fp);
	while(fgets(s, 1024, fp) != NULL){
		// 1st node --> node1
		tok = strtok(s, "\t|,");
		node1 = atoi(tok);
		// 2nd node --> node2
		tok = strtok(NULL, "\t|,");
		node2 = atoi(tok);
		if(node1 == node2) continue;
		// If both nodes exist, add the edge
		if(node_order[node1] < NodeNum && node_order[node2] < NodeNum){
            edges.push_back(make_pair(node_order[node1], node_order[node2]));
		}
	}
	fclose(fp);

    sort(edges.begin(),edges.end(), [](const pair<int, int>& a, const pair<int, int>& b){ return a.first < b.first; });

    fp = FileOpen(chosenEdgeFile, "w");
    for(i=0;i<edges.size();++i){
        j = edges[i].first;
        k = edges[i].second;
        fprintf(fp, to_string(j).c_str());
        fprintf(fp, "\t");
        fprintf(fp,to_string(k).c_str());
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
	int all_node_num;
	int triplet_num;
	int **node_order;
	map<int, int> *a_mat;			// adjacency matrix
	map<int, int>::iterator aitr;
	map<int, int>::iterator aitr2;
	int *deg;									// degree
	int itr;
	int i, j, k, x;
	string outdir;
	string outfile;
	char s[1025], *str;
	char str_1[] = "1";
	FILE *fp;

	// Initialization of Mersennne Twister
	unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4;
	init_by_array(init, length);

	EdgeFile = argv[1];

	NodeNum = -1;
	if (argc >= 3) NodeNum = atoi(argv[2]);
	printf("NodeNum is %d\n", NodeNum);
	ItrNum = 1;
	if (argc >= 4) ItrNum = atoi(argv[3]);
	printf("ItrNum is %d\n", ItrNum);

	// Total number of nodes --> all_node_num
	fp = FileOpen(EdgeFile, "r");
	for(i=0;i<2;i++) fgets(s, 1024, fp);
	for (i = 0; i < strlen(s) - 2; i++) {
        s[i] = s[i + 2];
    }
    s[i] = '\0';
	all_node_num = atoi(s);
	fclose(fp);
	printf("All node number is %d\n", all_node_num);

	// malloc
	malloc2D(&node_order, ItrNum, all_node_num);

	// Use all nodes
	if (NodeNum == -1){
		NodeNum = all_node_num;
		ItrNum = 1;
		for(j=0;j<NodeNum;j++) node_order[0][j] = j;
	}
	// Randomly generate the order of nodes --> node_order
	else{
		i = EdgeFile.find_last_of("/");
		outdir = EdgeFile.substr(0, i+1);
		outfile = outdir + "node-order_itr" + to_string(ItrNum) + ".csv";
		if(checkFileExistence(outfile)){
			fp = FileOpen(outfile, "r");
			for(j=0;j<all_node_num;j++){
				fgets(s, 1024, fp);
				strtok(s, ",");
				for(i=0;i<ItrNum;i++){
					node_order[i][j] = atoi(strtok(NULL, ","));
				}
			}
			fclose(fp);
		}
		else{
			for(i=0;i<ItrNum;i++){
				MakeRndPerm(node_order[i], all_node_num, all_node_num);
			}
			fp = FileOpen(outfile, "w");
			for(j=0;j<all_node_num;j++){
				fprintf(fp, "%d,", j);
				for(i=0;i<ItrNum;i++) fprintf(fp, "%d,", node_order[i][j]);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
	}

	// Output the header
	i = EdgeFile.find_last_of("/");
	outdir = EdgeFile.substr(0, i+1);

	// For each iteration
	for(itr=0;itr<ItrNum;itr++){
		// Initialization
        chosenEdgeFile = outdir + "outEdge_n" + to_string(NodeNum) + "_itr" + to_string(itr) + ".txt";
    	ReadEdges(node_order[itr]);
	}

	/************************* Output the results (AVG) *************************/
	// free
	free2D(node_order, ItrNum);

	return 0;
}
