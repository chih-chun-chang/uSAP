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
#include <set>

// Function to perform argsort
std::vector<int> argsort(std::vector<double> arr) {
  std::vector<int> sorted_indices(arr.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1, ..., arr.size()-1
  std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&arr](int i, int j){ return arr[i] < arr[j]; });  // Sort by arr values
  return sorted_indices;
}

std::vector<int> unique(std::vector<int> arr) {
  std::vector<int> unique_arr;
  std::set<int> seen;  // Used to track unique values
  for (auto elem : arr) {
    if (seen.find(elem) == seen.end()) {
      unique_arr.push_back(elem);
      seen.insert(elem);
    }
  }
  std::sort(unique_arr.begin(), unique_arr.end());  // Sort unique values
  return unique_arr;
}

// mergeFrom are not initially 0 (potential bugs??)
int carry_out_best_merges(std::vector<double> delta_entropy_for_each_block,
                          std::vector<int> best_merge_for_each_block,
                          std::vector<int>& b,
                          int B,
                          int B_to_merge
                         )
{
  std::vector<int> bestMerges = argsort(delta_entropy_for_each_block);
  std::vector<int> block_map(B);
  std::iota(block_map.begin(), block_map.end(), 0); 

  int num_merge = 0;
  int counter = 0;

  while (num_merge < B_to_merge) {
    int mergeFrom = bestMerges[counter];
    int mergeTo = block_map[best_merge_for_each_block[bestMerges[counter]]];
    /*
    int mergeTo;
    int idx = best_merge_for_each_block[bestMerges[counter]];
    if (idx == -1) {
      mergeTo = block_map.back();
    }
    else {
      mergeTo = block_map[idx];
    }
    */
    counter++;
    if (mergeTo != mergeFrom) {
      for (int i = 0; i < B; i++) {
        if (block_map[i] == mergeFrom) block_map[i] = mergeTo;
      }
      for (size_t i = 0; i < b.size(); i++) {
        if (b[i] == mergeFrom) b[i] = mergeTo;
      }
      num_merge += 1;
    }
  }

  std::vector<int> remaining_blocks = unique(b);
  std::vector<int> mapping(B, -1);
  for (size_t i = 0; i < remaining_blocks.size(); i++) {
    mapping[remaining_blocks[i]] = i;
  }
  
  std::vector<int> b_return;

  for (auto& it : b) {
    b_return.push_back(mapping[it]);
  }
  b = b_return;

  return B - B_to_merge;
}


void update_partition(int B,
                      std::vector<int>& b,
                      int ni,
                      int r,
                      int s,
                      std::vector<int>& M,
                      std::vector<int> M_r_row,
                      std::vector<int> M_s_row,
                      std::vector<int> M_r_col,
                      std::vector<int> M_s_col,
                      std::vector<int> d_out_new,
                      std::vector<int> d_in_new,
                      std::vector<int> d_new,
                      // for return
                      std::vector<int>& d_out,
                      std::vector<int>& d_in,
                      std::vector<int>& d
                     )
{

  b[ni] = s;
  for (int i = 0; i < B; i++) {
    M[r*B + i] = M_r_row[i];
    M[s*B + i] = M_s_row[i];
    M[i*B + r] = M_r_col[i];
    M[i*B + s] = M_s_col[i];
  }
  d_out = d_out_new;
  d_in = d_in_new;
  d = d_new;
}



bool prepare_for_partition_on_next_num_blocks(double S,
                                              std::vector<int>& b,
                                              std::vector<int>& M,
                                              std::vector<int>& d,
                                              std::vector<int>& d_out,
                                              std::vector<int>& d_in,
                                              int& B,
                                              int& B_to_merge,
                                              std::unordered_map< int, std::vector<int> >& old_b,
                                              std::unordered_map< int, std::vector<int> >& old_M,
                                              std::unordered_map< int, std::vector<int> >& old_d,
                                              std::unordered_map< int, std::vector<int> >& old_d_out,
                                              std::unordered_map< int, std::vector<int> >& old_d_in,
                                              std::unordered_map< int, double >& old_S,
                                              std::unordered_map< int, int >& old_B,
                                              double B_rate
                                             )

{

  bool optimal_B_found = false;
  int index;
  if (S <= old_S[1]) { // if the current partition is the best so far
    int old_index = old_B[1] > B ? 0 : 2;
    //  move the older
    old_b[old_index] = old_b[1];
    old_M[old_index] = old_M[1];
    old_d[old_index] = old_d[1];
    old_d_out[old_index] = old_d_out[1];
    old_d_in[old_index] = old_d_in[1];
    old_S[old_index] = old_S[1];
    old_B[old_index] = old_B[1];
    index = 1;
  }
  else { // the current partition is not the best so far
    index = old_B[1] > B ? 2 : 0; // if the current number of blocks is smaller than the best number of blocks so far
  }
   
  old_b[index] = b;
  old_M[index] = M;
  old_d[index] = d;
  old_d_out[index] = d_out;
  old_d_in[index] = d_in;
  old_S[index] = S;
  old_B[index] = B;


  // find the next number of blocks to try using golden ratio bisection
  if (std::isinf(old_S[2])) {
    B_to_merge = (int)B*B_rate;
    if (B_to_merge == 0) optimal_B_found = true;
    b = old_b[1];
    M = old_M[1];
    d = old_d[1];
    d_out = old_d_out[1];
    d_in = old_d_in[1];
  }
  else {
    // golden ratio search bracket established
    if (old_B[0] - old_B[2] == 2) { // we have found the partition with the optimal number of blocks
      optimal_B_found = true;
      B = old_B[1];
      b = old_b[1];
    }
    else { // not done yet, find the next number of block to try according to the golden ratio search
        if ((old_B[0]-old_B[1]) >= (old_B[1]-old_B[2])) {  // the higher segment in the bracket is bigger
          index = 0;
        }
        else {
          index = 1;
        }
        int next_B_to_try = old_B[index + 1] + (int)(old_B[index] - old_B[index + 1]) * 0.618;
        B_to_merge = old_B[index] - next_B_to_try;
        B = old_B[index];
        b = old_b[index];
        M = old_M[index];
        d = old_d[index];
        d_out = old_d_out[index];
        d_in = old_d_in[index];
      }
    }
  return optimal_B_found;
}




