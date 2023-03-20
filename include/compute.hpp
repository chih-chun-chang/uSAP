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

void compute_new_rows_cols_interblock_edge_count_matrix(int B,
                                                        std::vector<int> M,
                                                        int r,
                                                        int s,
                                                        std::vector<int> b_out,
                                                        std::vector<int> count_out,
                                                        std::vector<int> b_in,
                                                        std::vector<int> count_in,
                                                        int count_self,
                                                        int agg_move,
                                                        // for return
                                                        std::vector<int>& M_r_row,
                                                        std::vector<int>& M_s_row,
                                                        std::vector<int>& M_r_col,
                                                        std::vector<int>& M_s_col
                                                       )
{
  std::vector<int> M_r_row_temp(B, 0);
  std::vector<int> M_r_col_temp(B, 0);
  std::vector<int> M_s_row_temp(B, 0);
  std::vector<int> M_s_col_temp(B, 0);
  
  if (agg_move) { // the r row and column are simply empty after this merge move
    M_r_row = M_r_row_temp;
    M_r_col = M_r_col_temp; 
  }
  else {

    // copy M[r, :] to M_r_row
    // copy M[:, r] to M_r_col
    for (int i = 0; i < B; i++) {
      M_r_row_temp[i] = M[r*B + i];  
      M_r_col_temp[i] = M[i*B + r];
    }

    // check b_out.size() == count_out.size()    
    for (size_t i = 0; i < b_out.size(); i++) {
      M_r_row_temp[ b_out[i] ] -= count_out[i];
    }  
   
    // b_in is unique 
    int count_in_sum_r = 0;
    for (size_t i = 0; i < b_in.size(); i++) {
      if (b_in[i] == r) {
        count_in_sum_r += count_in[i];
      }
    }
    M_r_row_temp[r] -= count_in_sum_r;
    M_r_row_temp[s] += count_in_sum_r;

    // check b_in.size() == count_in.size()
    for (size_t i = 0; i < b_in.size(); i++) {
      M_r_col_temp[ b_in[i] ] -= count_in[i];
    }

    // b_out is unique
    int count_out_sum_r = 0;
    for (size_t i = 0; i < b_out.size(); i++) {
      if (b_out[i] == r) {
        count_out_sum_r += count_out[i];
      }
    }
    
    M_r_col_temp[r] -= count_out_sum_r;
    M_r_col_temp[s] += count_out_sum_r;
  }
  
  // copy M[s, :] to M_s_row
  // copy M[:, s] to M_s_col
  for (int i = 0; i < B; i++) {
    M_s_row_temp[i] = M[s*B + i];
    M_s_col_temp[i] = M[i*B + s];
  }

  // check b_out.size() == count_out.size()  
  for (size_t i = 0; i < b_out.size(); i++) {
    M_s_row_temp[ b_out[i] ] += count_out[i];
  }

  int count_in_sum_s = 0;
  for (size_t i = 0; i < b_in.size(); i++) {
    if (b_in[i] == s) {
      count_in_sum_s += count_in[i];
    } 
  }
  M_s_row_temp[r] -= count_in_sum_s;
  M_s_row_temp[s] += count_in_sum_s; 
  M_s_row_temp[r] -= count_self;
  M_s_row_temp[s] += count_self;

  for (size_t i = 0; i < b_in.size(); i++) {
    M_s_col_temp[ b_in[i] ] += count_in[i];
  }

  int count_out_sum_s = 0;
  for (size_t i = 0; i < b_out.size(); i++) {
    if (b_out[i] == s) {
      count_out_sum_s += count_out[i];
    } 
  }
  M_s_col_temp[r] -= count_out_sum_s;
  M_s_col_temp[s] += count_out_sum_s;
  M_s_col_temp[r] -= count_self;
  M_s_col_temp[s] += count_self;

  M_r_row = M_r_row_temp;
  M_r_col = M_r_col_temp;
  M_s_row = M_s_row_temp;
  M_s_col = M_s_col_temp;
}

void compute_new_block_degree(int r,
                              int s,
                              std::vector<int> d_out,
                              std::vector<int> d_in,
                              std::vector<int> d,
                              int k_out,
                              int k_in,
                              int k,
                              // for return
                              std::vector<int>& d_out_new,
                              std::vector<int>& d_in_new,
                              std::vector<int>& d_new)
{


  d_out[r] -= k_out;
  d_out[s] += k_out;

  d_in[r] -= k_in;
  d_in[s] += k_in;

  d[r] -= k;
  d[s] += k;

  d_out_new = d_out;
  d_in_new = d_in;
  d_new = d;

}


double compute_Hastings_correction(std::vector<int> b_out,
                                   std::vector<int> count_out,
                                   std::vector<int> b_in,
                                   std::vector<int> count_in,
                                   int s,
                                   std::vector<int> M,
                                   std::vector<int> M_r_row,
                                   std::vector<int> M_r_col,
                                   int B,
                                   std::vector<int> d,
                                   std::vector<int> d_new
                                  )
{
  std::map<int, int> map;
  for (size_t i = 0; i < b_out.size(); i++) {
    map[b_out[i]] += count_out[i];
  }
  for (size_t i = 0; i < b_in.size(); i++) {
    map[b_in[i]] += count_in[i];
  }

  std::vector<int> t;
  std::vector<int> count;
  for (const auto& [key, value] : map) {
    t.push_back(key);
    count.push_back(value);
  }

  std::vector<int> M_t_s;
  std::vector<int> M_s_t;
  for (auto& tt : t) {
    M_t_s.push_back(M[tt*B + s]);
    M_s_t.push_back(M[s*B + tt]);
  }

  double p_forward = 0;
  double p_backward = 0;
  for (size_t i = 0; i < t.size(); i++) {
    p_forward += (double)count[i] * (M_t_s[i] + M_s_t[i] + 1) / (d[t[i]] + B);
    p_backward += (double)count[i] * (M_r_row[t[i]] + M_r_col[t[i]] + 1) / (d_new[t[i]] + B);
  }

  return p_backward / p_forward;
}


double compute_delta_entropy(int B,
                             int r,
                             int s,
                             std::vector<int> M,
                             std::vector<int> M_r_row,
                             std::vector<int> M_s_row,
                             std::vector<int> M_r_col,
                             std::vector<int> M_s_col,
                             std::vector<int> d_out,
                             std::vector<int> d_in,
                             std::vector<int> d_out_new,
                             std::vector<int> d_in_new
                             )
{

  std::vector<int> M_r_t1(B, 0);
  std::vector<int> M_s_t1(B, 0);
  std::vector<int> M_t2_r(B, 0);
  std::vector<int> M_t2_s(B, 0);

  for (int i = 0; i < B; i++) {
    M_r_t1[i] = M[r*B + i];
    M_s_t1[i] = M[s*B + i];
    M_t2_r[i] = M[i*B + r];
    M_t2_s[i] = M[i*B + s];
  }

  // remove r and s from the cols to avoid double counting
  std::vector<int> M_r_col_tmp;
  for (int i = 0; i < M_r_col.size(); i++) {
    if (i != r && i != s)
      M_r_col_tmp.push_back(M_r_col[i]);
  }
  M_r_col = M_r_col_tmp;

  std::vector<int> M_s_col_tmp;
  for (int i = 0; i < M_s_col.size(); i++) {
    if (i != r && i != s)
      M_s_col_tmp.push_back(M_s_col[i]);
  }
  M_s_col = M_s_col_tmp;

  std::vector<int> M_t2_r_tmp;
  for (int i = 0; i < M_t2_r.size(); i++) {
    if (i != r && i != s)
      M_t2_r_tmp.push_back(M_t2_r[i]);
  }
  M_t2_r = M_t2_r_tmp;

  std::vector<int> M_t2_s_tmp;
  for (int i = 0; i < M_t2_s.size(); i++) {
    if (i != r && i != s)
      M_t2_s_tmp.push_back(M_t2_s[i]);
  }
  M_t2_s = M_t2_s_tmp;


  std::vector<int> d_out_new_;
  std::vector<int> d_out_;
  for (int i = 0; i < d_out_new.size(); i++) {
    if (i != r && i != s)
      d_out_new_.push_back(d_out_new[i]);
  }
  for (int i = 0; i < d_out.size(); i++) {
    if (i != r && i != s)
      d_out_.push_back(d_out[i]);
  }
  
  // only keep non-zero entries to avoid unnecessary computation
  std::vector<int> d_in_new_r_row;
  std::vector<int> d_in_new_s_row;
  std::vector<int> M_r_row_non_zero;
  std::vector<int> M_s_row_non_zero;
  std::vector<int> d_out_new_r_col;
  std::vector<int> d_out_new_s_col;
  std::vector<int> M_r_col_non_zero;
  std::vector<int> M_s_col_non_zero;
  std::vector<int> d_in_r_t1;
  std::vector<int> d_in_s_t1;
  std::vector<int> M_r_t1_non_zero;
  std::vector<int> M_s_t1_non_zero;
  std::vector<int> d_out_r_col;
  std::vector<int> d_out_s_col;
  std::vector<int> M_t2_r_non_zero;
  std::vector<int> M_t2_s_non_zero;

  //printf("%d %d\n", B, M_r_row.size());
  for (size_t i = 0; i < M_r_row.size(); i++) {
    if (M_r_row[i] != 0) { // nonzero index
      d_in_new_r_row.push_back(d_in_new[i]);
      M_r_row_non_zero.push_back(M_r_row[i]);
    }
  }

  for (size_t i = 0; i < M_s_row.size(); i++) {
    if (M_s_row[i] != 0) {
      d_in_new_s_row.push_back(d_in_new[i]);
      M_s_row_non_zero.push_back(M_s_row[i]);
    }
  }

  for (size_t i = 0; i < M_r_t1.size(); i++) {
    if (M_r_t1[i] != 0) {
      d_in_r_t1.push_back(d_in[i]);
      M_r_t1_non_zero.push_back(M_r_t1[i]);
    }
  }

  for (size_t i = 0; i < M_s_t1.size(); i++) {
    if (M_s_t1[i] != 0) {
      d_in_s_t1.push_back(d_in[i]);
      M_s_t1_non_zero.push_back(M_s_t1[i]);
    }
  }

  for (size_t i = 0; i < M_r_col.size(); i++) {
    if (M_r_col[i] != 0) {
      d_out_new_r_col.push_back(d_out_new_[i]);
      M_r_col_non_zero.push_back(M_r_col[i]);
    }   
  }

  for (size_t i = 0; i < M_s_col.size(); i++) {
    if (M_s_col[i] != 0) {
      d_out_new_s_col.push_back(d_out_new_[i]);
      M_s_col_non_zero.push_back(M_s_col[i]);
    }
  }

  for (size_t i = 0; i < M_t2_r.size(); i++) {
    if (M_t2_r[i] != 0) {
      d_out_r_col.push_back(d_out_[i]); 
      M_t2_r_non_zero.push_back(M_t2_r[i]);
    }
  }

  for (size_t i = 0; i < M_t2_s.size(); i++) {
    if (M_t2_s[i] != 0) {
      d_out_s_col.push_back(d_out_[i]);
      M_t2_s_non_zero.push_back(M_t2_s[i]);
    } 
  }

  // sum over the two changed rows and cols
  double delta_entropy = 0;
  // bug here? 
  // (may enter the loop cause nan)
  // (may contain 0 in the vector "sometimes")
  double tmp = 0;
  for (size_t i = 0; i < M_r_row_non_zero.size(); i++) {
    //if (d_in_new_r_row[i] != 0 && d_out_new[r] != 0) {
      double temp = (double)M_r_row_non_zero[i]/d_in_new_r_row[i]/d_out_new[r];
      delta_entropy -= (double)M_r_row_non_zero[i]*log(temp);
      tmp += (double)M_r_row_non_zero[i]*log(temp);
    //}
  }

  //printf("M_r_row: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_s_row_non_zero.size(); i++) {
    //if (d_in_new_s_row[i] != 0 && d_out_new[s] != 0) {
      double temp = (double)M_s_row_non_zero[i]/d_in_new_s_row[i]/d_out_new[s];
      delta_entropy -= (double)M_s_row_non_zero[i]*log(temp);
      tmp += (double)M_s_row_non_zero[i]*log(temp);
    //}
  }

  //printf("M_s_row: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_r_col_non_zero.size(); i++) {
    //if (d_out_new_r_col[i] != 0 && d_in_new[r] != 0) {    
      double temp = (double)M_r_col_non_zero[i]/d_out_new_r_col[i]/d_in_new[r];
      delta_entropy -= (double)M_r_col_non_zero[i]*log(temp);
      tmp += (double)M_r_col_non_zero[i]*log(temp);
    //}
  }

  //printf("M_r_col: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_s_col_non_zero.size(); i++) {
    //if (d_out_new_s_col[i] != 0 && d_in_new[s] != 0) {    
      double temp = (double)M_s_col_non_zero[i]/d_out_new_s_col[i]/d_in_new[s];
      delta_entropy -= (double)M_s_col_non_zero[i]*log(temp);
      tmp += (double)M_s_col_non_zero[i]*log(temp); 
    //}
  }

  //printf("M_s_col: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_r_t1_non_zero.size(); i++) {
    //if (d_in_r_t1[i] != 0 && d_out[r] != 0) {    
      double temp = (double)M_r_t1_non_zero[i]/d_in_r_t1[i]/d_out[r];
      delta_entropy += (double)M_r_t1_non_zero[i]*log(temp);
      tmp += (double)M_r_t1_non_zero[i]*log(temp);
    //}
  }

  //printf("M_r_t1: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_s_t1_non_zero.size(); i++) {
    //if (d_in_s_t1[i] != 0 && d_out[s] != 0) {    
      double temp = (double)M_s_t1_non_zero[i]/d_in_s_t1[i]/d_out[s];
      delta_entropy += (double)M_s_t1_non_zero[i]*log(temp);
      tmp += (double)M_s_t1_non_zero[i]*log(temp);
    //}
  }

  //printf("M_s_t1: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_t2_r_non_zero.size(); i++) {
    //if (d_out_r_col[i] != 0 && d_in[r] != 0) {
      double temp = (double)M_t2_r_non_zero[i]/d_out_r_col[i]/d_in[r];
      // some 0 in the end of d_out_r_col
      delta_entropy += (double)M_t2_r_non_zero[i]*log(temp);
      tmp += (double)M_t2_r_non_zero[i]*log(temp);
    //}
  }

  //printf("M_t2_r: %f\n", tmp);
  tmp = 0;

  for (size_t i = 0; i < M_t2_s_non_zero.size(); i++) {
    //if (d_out_s_col[i] != 0 && d_in[s] != 0) {
      double temp = (double)M_t2_s_non_zero[i]/d_out_s_col[i]/d_in[s];
      delta_entropy += (double)M_t2_s_non_zero[i]*log(temp);
      tmp += (double)M_t2_s_non_zero[i]*log(temp);
    //}
  }

  //printf("M_t2_r: %f\n", tmp);
  tmp = 0;

  return delta_entropy;
}

double compute_overall_entropy(std::vector<int> M,
                               std::vector<int> d_out,
                               std::vector<int> d_in,
                               int B,
                               int N,
                               int E)
{
  std::vector<int> nonzero_row;
  std::vector<int> nonzero_col;
  std::vector<int> edge_count_entries;
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      if (M[i*B + j] != 0) {
        nonzero_row.push_back(i);
        nonzero_col.push_back(j);
        edge_count_entries.push_back(M[i*B + j]);
      }
    }
  }

  double data_S = 0;
  for (size_t i = 0; i < edge_count_entries.size(); i++) {
    data_S -= edge_count_entries[i] * log(edge_count_entries[i] / (double)(d_out[nonzero_row[i]] * d_in[nonzero_col[i]]));
  }

  double model_S_term = (double)B*B/E;
  double model_S = (double)(E * (1 + model_S_term) * log(1 + model_S_term)) - (model_S_term * log(model_S_term)) + (N * log(B)); 
  
  return model_S + data_S;

}




