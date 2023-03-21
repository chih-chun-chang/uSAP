#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

#include "graph.hpp"


// TODO: Graph::_initialize_edge_count
void initialize_edge_counts(
  
  const std::vector<std::...>& ...
  // this is fucking stupid --- please avoid in the future
  //std::unordered_map< int, std::vector<std::vector<int>> > out_neighbors,
                            int B,
                            std::vector<int> b,
                            // for return
                            std::vector<int>& M,
                            std::vector<int>& d_out,
                            std::vector<int>& d_in,
                            std::vector<int>& d
                           )
{

  M.clear();
  M.resize(B*B, 0); // initialize to zero

  // compute the initial interblock edge count
  for (const auto& [v, neighbors] : out_neighbors) {
    if (neighbors.size() > 0) {
      int k1 = b[v];
      std::map<int, int> out;
      for (auto& n: neighbors) {
        int key = b[n[0]];
        out[key] += n[1];
        //M[k1*B + k2] += n[1]; 
      }
      for (const auto& [k2, count] : out) {
        M[k1*B + k2] += count;
      }
    }
  }
  // compute initial block degrees
  d_out.clear();
  d_out.resize(B, 0);
  d_in.clear();
  d_in.resize(B, 0);
  d.clear();
  d.resize(B, 0);
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      d_out[i] += M[i*B + j]; // row
      d_in[i] += M[j*B + i]; // col
    }
    d[i] = d_out[i] + d_in[i];
  }
}

