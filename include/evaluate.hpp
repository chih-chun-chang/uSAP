// https://github.com/hpec-graphchallenge/BlockFinder/blob/master/blockfinder/evaluate.hpp
//
//

#pragma once

#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

#include "km.hpp"

namespace bf {

// Procedure: evaluate
// Evaluate the quality of a given partition to a true partition
void evaluate(const std::vector<int> &truth, const std::vector<int> &given) {

  assert(truth.size() == given.size());

  std::unordered_set<int> blocks_b1_set(truth.begin(), truth.end());
  blocks_b1_set.erase(-1);  // -1 is the label for 'unknown'
  int B_b1 = blocks_b1_set.size();
  int B_b2 = *std::max_element(given.begin(), given.end()) + 1;

  std::cout << "========Partition Correctness Evaluation========\n";
  std::cout << "Number of nodes: "<< given.size() << '\n';
  std::cout << "Number of partitions (truth): " << B_b1 <<"\n";
  std::cout << "Number of partitions (given): " << B_b2 <<"\n";

  // populate the confusion matrix between the two partitions
  std::vector<std::vector<int>> contingency_table(B_b1, std::vector<int>(B_b2,0));

  // evaluation is based on nodes observed so far
  int N = 0;
  for(size_t i=0;i<given.size();i++) {
    // do not include nodes without truth
    if(truth[i] != -1) {
      contingency_table[truth[i]][given[i]] += 1;
      N+=1;
    }
  }

}

}  // end of namespace bf. ----------------------------------------------------
