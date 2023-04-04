#pragma once

#include <vector>
#include <set>
#include <algorithm>

namespace sgp {

std::vector<size_t> unique(std::vector<size_t> arr) {
  std::vector<size_t> unique_arr;
  std::set<size_t> seen;  // Used to track unique values
  for (auto elem : arr) {
    if (seen.find(elem) == seen.end()) {
      unique_arr.push_back(elem);
      seen.insert(elem);
    }
  }
  std::sort(unique_arr.begin(), unique_arr.end());  // Sort unique values
  return unique_arr;
}

std::vector<size_t> argsort(std::vector<float> arr) {
  std::vector<size_t> sorted_indices(arr.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1, ..., arr.size()-1
  std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&arr](int i, int j){ return arr[i] < arr[j]; });  // Sort by arr values
  return sorted_indices;
}


}