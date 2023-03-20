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

void propose_new_partition(int r,
                           std::vector< std::vector<int> > neighbors_out,
                           std::vector< std::vector<int> > neighbors_in,
                           std::vector<int> b,
                           std::vector<int> M,
                           std::vector<int> d,
                           int B,
                           int agg_move,
                           // for return
                           int& s,
                           int& k_out,
                           int& k_in,
                           int& k
                          )
{
  
  std::vector< std::vector<int> > neighbors(neighbors_out);
  neighbors.insert(neighbors.end(), neighbors_in.begin(), neighbors_in.end());

  k_out = 0;
  for (auto& it : neighbors_out) k_out += it[1];
  k_in = 0;
  for (auto& it : neighbors_in) k_in += it[1];
  k = k_out + k_in;

  std::random_device rd; // seed the random number generator
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> randint(0, B-1); // 0 ~ B-1

  if (k == 0) { // this node has no neighbor, simply propose a block randomly
    s = randint(gen);
    return;
  }

  // create the probabilities array based on the edge weight of each neighbor
  std::vector<double> probabilities;
  for (auto& n : neighbors) {
    probabilities.push_back( (double)n[1]/k );
  }
  // create a discrete distribution based on the probabilities array
  std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
  int rand_neighbor = neighbors[distribution(gen)][0];
  int u = b[rand_neighbor];

  // propose a new block randomly
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
  double rand_num = uni_dist(gen);
  if ( rand_num <= (double)B/(d[u]+B) ) { // chance inversely prop. to block_degree
    if (agg_move) { // force proposal to be different from current block
      std::vector<int> candidates;
      for (int i = 0; i < B; i++) {
        if (i != r) candidates.push_back(i);
      }
      std::uniform_int_distribution<int> choice(0, candidates.size()-1);
      s = candidates[choice(gen)];
    }
    else{
      s = randint(gen);
    }
  }
  else { // propose by random draw from neighbors of block partition[rand_neighbor]
    std::vector<double> multinomial_prob(B);
    double multinomial_prob_sum = 0;
    for (int i = 0; i < B; i++) {
      multinomial_prob[i] = (double)(M[u*B + i] + M[i*B + u])/d[u];
      multinomial_prob_sum += multinomial_prob[i];
    }
    if (agg_move) { // force proposal to be different from current block
      multinomial_prob[r] = 0;
      // recalculate
      multinomial_prob_sum = 0;
      for (int i = 0; i < B; i++) {
        multinomial_prob_sum += multinomial_prob[i];
      }
      // check
      if (multinomial_prob_sum == 0) { // the current block has no neighbors. randomly propose a different block
        std::vector<int> candidates;
        for (int i = 0; i < B; i++) {
          if (i != r) candidates.push_back(i);
        }
        std::uniform_int_distribution<int> choice(0, candidates.size()-1);
        s = candidates[choice(gen)];
        return;
      }
      else {
        for (auto& it : multinomial_prob) it /= multinomial_prob_sum;
      }
    }
    std::vector<int> nonzero_index;
    std::vector<double> nonzero_prob;
    for (int i = 0; i < B; i++) {
      if (multinomial_prob[i] != 0) {
        nonzero_index.push_back(i);
        nonzero_prob.push_back(multinomial_prob[i]);
      }
    }
    std::discrete_distribution<int> multinomial(nonzero_prob.begin(), nonzero_prob.end());
    int cand = multinomial(gen);
    s = nonzero_index[cand];
  }
}
                          
