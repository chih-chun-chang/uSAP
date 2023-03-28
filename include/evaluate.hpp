// https://github.com/hpec-graphchallenge/BlockFinder/blob/master/blockfinder/evaluate.hpp
//
#pragma once

#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

#include "km.hpp"

namespace bf {

// Procedure: evaluate
// Evaluate the quality of a given partition to a true partition
template <typename V>
void evaluate(const std::vector<V> &truth, const std::vector<V> &given) {

  assert(truth.size() == given.size());

  std::unordered_set<V> blocks_b1_set(truth.begin(), truth.end());
  blocks_b1_set.erase(-1);  // -1 is the label for 'unknown'
  V B_b1 = blocks_b1_set.size();
  V B_b2 = *std::max_element(given.begin(), given.end()) + 1;

  std::cout << "========Partition Correctness Evaluation========\n";
  std::cout << "Number of nodes: "<< given.size() << '\n';
  std::cout << "Number of partitions (truth): " << B_b1 <<"\n";
  std::cout << "Number of partitions (given): " << B_b2 <<"\n";

  // populate the confusion matrix between the two partitions
  std::vector<std::vector<V>> contingency_table(B_b1, std::vector<V>(B_b2,0));
  
  // evaluation is based on nodes observed so far
  V N = 0;
  for(size_t i=0;i<given.size();i++) {
    // do not include nodes without truth
    if(truth[i] != -1) {
      contingency_table[truth[i]][given[i]] += 1;
      N+=1;
    }
  }

  // associate the labels between two partitions using linear assignment
  // use the Hungarian algorithm / Kuhn-Munkres algorithm
  std::vector<std::vector<V>> tmp;
  // transpose matrix for linear assignment (this implementation assumes #col >= #row)
  if(B_b1 > B_b2) {  
    tmp = std::vector<std::vector<V>>(B_b2, std::vector<V>(B_b1,0));
    for(size_t i=0;i<contingency_table.size();i++) {
      for(size_t j=0;j<contingency_table[i].size();j++) {
        tmp[j][i] = contingency_table[i][j];
      }
    }
  }
  else {
    tmp = contingency_table;
  }
  
  // K-M assignment
  bf::KuhnMunkresSolver<V> km;
  km.solve(tmp);
  auto& indices = km.match1();

  V total = 0;
  std::vector<std::vector<V>> contingency_table_before_assignment = tmp;
  for(size_t i=0;i<indices.size();i++) {
    V row=i;
    V col=indices[i];
    for(size_t j=0;j<contingency_table_before_assignment.size();j++) {
      tmp[j][row] = contingency_table_before_assignment[j][col];
    }
    total += tmp[row][row];   
  }
  
  // fill in the un-associated columns
  std::vector<bool> unassociated_col(tmp[0].size(), true);
  for(size_t i=0;i<tmp.size();i++) {
    unassociated_col[indices[i]] = false;
  }

  V counter = 0;
  for(size_t column = 0; column<tmp.size(); column++) {
    if(unassociated_col[column] == false) {
      continue;
    }
    for(size_t j=0;j<tmp.size();j++) {
      tmp[j][tmp.size() + counter] = contingency_table_before_assignment[j][column];
    }
    counter += 1;
  }

  // transpose back
  if(B_b1 > B_b2) { 
    for(size_t i=0;i<contingency_table.size();i++) {
      for(size_t j=0;j<contingency_table[i].size();j++) {
        contingency_table[i][j] = tmp[j][i];
      }
    }
  }
  else {
    contingency_table = tmp;
  }

  // joint probability of the two partitions is just the normalized contingency table
  std::vector<std::vector<float>> joint_prob(B_b1, std::vector<float>(B_b2, 0.0f));

  auto accuracy = 0.0f;
  
  for(size_t i=0;i<contingency_table.size(); i++) {
    for(size_t j=0;j<contingency_table[i].size();j++) {
      std::cout << contingency_table[i][j] << " ";
      joint_prob[i][j] = contingency_table[i][j]/(float)N;
    }
    std::cout << '\n';
    accuracy += joint_prob[i][i];
  }
  std::cout << "Accuracy (with optimal partition matching): " << accuracy << '\n';
  std::cout << '\n';

  // Compute pair-counting-based metrics
  V num_pairs = N*(N-1)/2;
  std::vector<V> colsum(contingency_table[0].size(),0), rowsum(contingency_table.size(),0);
  V sum_table_squared = 0, num_agreement_same = 0;
  for(size_t i=0;i<contingency_table.size();i++) {
    for(size_t j=0;j<contingency_table[i].size();j++) {
      colsum[j] += contingency_table[i][j];
      rowsum[i] += contingency_table[i][j];
      sum_table_squared += contingency_table[i][j] * contingency_table[i][j];
      num_agreement_same += (contingency_table[i][j] * (contingency_table[i][j]-1))/2;
    }
  }

  // compute counts of agreements and disagreement (4 types) and the regular rand index
  V sum_colsum_squared = 0;
  for(size_t i=0;i<colsum.size();i++) {
    sum_colsum_squared += colsum[i]*colsum[i];
  }

  V sum_rowsum_squared = 0;
  for(size_t i=0;i<rowsum.size();i++) {
    sum_rowsum_squared +=  rowsum[i]*rowsum[i];
  }

  std::vector<V> count_in_each_b1 = rowsum;
  std::vector<V> count_in_each_b2 = colsum;

  V num_same_in_b1 = 0;
  for(size_t i=0;i<count_in_each_b1.size();i++) {
    num_same_in_b1 += (count_in_each_b1[i] * (count_in_each_b1[i] - 1)) / 2;
  }

  V num_same_in_b2 = 0;
  for(size_t i=0;i<count_in_each_b2.size();i++) {
    num_same_in_b2 += (count_in_each_b2[i] * (count_in_each_b2[i] - 1)) / 2;
  }

  auto num_agreement_diff = (N*N + sum_table_squared - sum_colsum_squared - sum_rowsum_squared)/2.0f;
  auto num_agreement = num_agreement_same + num_agreement_diff;
  auto rand_index = num_agreement / num_pairs;

  V sum_table_choose_2 = num_agreement_same;
  V sum_colsum_choose_2 = num_same_in_b2;
  V sum_rowsum_choose_2 = num_same_in_b1;
  auto adjusted_rand_index = (sum_table_choose_2 - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs) / (
  0.5f * (sum_rowsum_choose_2 + sum_colsum_choose_2) - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs);

  std::cout << "Rand Index: " << rand_index << '\n';
  std::cout << "Adjusted Rand Index: " << adjusted_rand_index << '\n';
  std::cout << "Pairwise Recall: " << (float)num_agreement_same / num_same_in_b1 << '\n';
  std::cout << "Pairwise Precision: " << (float)num_agreement_same / num_same_in_b2 << '\n';
  std::cout << '\n';

  // compute the information theoretic metrics
  std::vector marginal_prob_b2(contingency_table[0].size(), 0.0f);
  std::vector marginal_prob_b1(contingency_table.size(), 0.0f);
  
  for(size_t i=0;i<marginal_prob_b2.size();i++) {
    marginal_prob_b2[i] = colsum[i]/(float)N;
  }

  for(size_t i=0;i<marginal_prob_b1.size();i++) {
    marginal_prob_b1[i] = rowsum[i]/(float)N;
  }

  std::vector<std::vector<float>> conditional_prob_b2_b1(
    joint_prob.size(), std::vector<float>(joint_prob[0].size(),0.0f)
  );

  std::vector<std::vector<float>> conditional_prob_b1_b2(
    joint_prob.size(), std::vector<float>(joint_prob[0].size(),0.0f)
  );

  auto H_b2=0.0f, H_b1=0.0f, H_b2_b1=0.0f, H_b1_b2=0.0f;

  // compute entropy of the non-partition2 and the partition2 version
  for(size_t i=0;i<marginal_prob_b1.size();i++){
    if(marginal_prob_b1[i]==0) {
      continue;
    }
    for(size_t j=0;j<joint_prob[i].size();j++) {
      conditional_prob_b2_b1[i][j] = joint_prob[i][j]/marginal_prob_b1[i];
    }
    H_b1 -= marginal_prob_b1[i] * std::log(marginal_prob_b1[i]);
  }

  for(size_t i=0;i<marginal_prob_b2.size();i++){
    if(marginal_prob_b2[i]==0) {
      continue;
    }
    for(size_t j=0;j<joint_prob.size();j++) {
      conditional_prob_b1_b2[j][i] = joint_prob[j][i]/marginal_prob_b2[i];
    }
    H_b2 -= marginal_prob_b2[i] * std::log(marginal_prob_b2[i]);
  }

  // compute the conditional entropies
  for(size_t i=0;i<joint_prob.size();i++){
    for(size_t j=0;j<joint_prob[i].size();j++){
      if(joint_prob[i][j]!=0){
        H_b2_b1 -= joint_prob[i][j] * std::log(conditional_prob_b2_b1[i][j]);
        H_b1_b2 -= joint_prob[i][j] * std::log(conditional_prob_b1_b2[i][j]);
      }
    }
  }

  // compute the mutual information (symmetric)
  auto MI_b1_b2 = 0.0f;
  for(size_t i=0;i<marginal_prob_b1.size();i++) {
    for(size_t j=0;j<marginal_prob_b2.size();j++) {
      if(joint_prob[i][j]!=0) {
        MI_b1_b2 += joint_prob[i][j] * 
                    std::log(joint_prob[i][j] / (marginal_prob_b1[i]*marginal_prob_b2[j]));
      }
    }
  }

  auto fraction_missed_info = 0.0f, fraction_err_info = 0.0f;

  if(H_b1 > 0.0f) {
    fraction_missed_info = H_b1_b2 / H_b1;
  }

  if(H_b2 > 0.0f) {
    fraction_err_info = H_b2_b1 / H_b2;
  }

  std::cout << "Entropy of truth partition: " << std::abs(H_b1) << '\n';
  std::cout << "Entropy of given partition: " << std::abs(H_b2) << '\n';
  std::cout << "Conditional entropy of truth under given: " << std::abs(H_b1_b2) << '\n';
  std::cout << "Conditional entropy of given under truth: " << std::abs(H_b2_b1) << '\n';
  std::cout << "Mututal info between truth and given: " << std::abs(MI_b1_b2) << '\n';
  std::cout << "Fraction of missed info: " << std::abs(fraction_missed_info) << '\n';
  std::cout << "Fraction of erroneous info: " << std::abs(fraction_err_info) << '\n';
  std::cout << "================================================\n";
}

}  // end of namespace bf. ----------------------------------------------------//
