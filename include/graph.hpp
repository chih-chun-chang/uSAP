#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>


class Graph {
  public:
    int N; // number of node
    int E; // number of edge

    std::unordered_map< int, std::unordered_map<int, int> > adjList; // adjList[i][j] = w_ij (i->j)
    std::unordered_map< int, std::vector<std::vector<int>> > in_neighbors; // {in_neighbor_index}
    std::unordered_map< int, std::vector<std::vector<int>> > out_neighbors; // {out_neighbor_index}

    std::vector<int> true_partition; // (node, block)


    Graph() : N(0), E(0){} // constructor

    void load_graph_from_tsv(std::string FileName) { 
      
      std::ifstream file(FileName + ".tsv"); // open the file in read mode
      if (!file.is_open()) {
        std::cerr << "Unable to open file!\n";
        exit(1);
      }
      
      std::string line;
      // format: node i \t node j \t  w_ij
      while (getline(file, line)) {
        size_t start = 0;
        size_t tab_pos = line.find('\t');
        int i = std::stoi(line.substr(start, tab_pos - start));
        start = tab_pos + 1;
        tab_pos = line.find('\t', start);
        int j = std::stoi(line.substr(start, tab_pos - start));
        start = tab_pos + 1;
        tab_pos = line.find('\t', start);
        int w_ij = std::stoi(line.substr(start, tab_pos - start));

        // NOTE: index start from 1
        adjList[i-1][j-1] = w_ij; // e(i, j) = w_ij
        
        std::vector<int> out{j-1, w_ij};
        std::vector<int> in{i-1, w_ij};
        out_neighbors[i-1].push_back(out);
        in_neighbors[j-1].push_back(in);

        // count the vertices and edges
        if (i > N) N = i;
          E++;
      }
      file.close();

      // load the true partition
      std::ifstream true_file(FileName + "_truePartition.tsv");
      if (!true_file.is_open()) {
        std::cerr << "Unable to open file!\n";
        return;
      }
      // format: node i \t block
      while (getline(true_file, line)) {
        size_t start = 0;
        size_t tab_pos = line.find('\t');
        int i = std::stoi(line.substr(start, tab_pos - start));
        start = tab_pos + 1;
        tab_pos = line.find('\t', start);
        int block = std::stoi(line.substr(start, tab_pos - start));
        true_partition.push_back(block);
      }
      true_file.close();
    } 

};


