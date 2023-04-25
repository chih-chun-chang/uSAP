#pragma once 
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <cassert>
#include <set>
#include <thread>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/reduce.hpp>

namespace sgp {

template <typename W>
class Graph_P {

  public: 

    struct Edge {
      size_t from;
      size_t to;
      W weight;
    };

    // parameters can be set by users
    int beta = 3;
    size_t num_agg_proposals_per_block = 10; 
    float num_block_reduction_rate = 0.5;
    size_t max_num_nodal_itr = 100;
    float delta_entropy_threshold1 = 5e-4;
    float delta_entropy_threshold2 = 1e-4;
    size_t delta_entropy_moving_avg_window = 3;
    bool verbose = true;

    // function used by users
    void load_graph_from_tsv(const std::string& FileName);
    std::vector<size_t> partition();    
    const size_t& num_nodes() const { return _N; }
    const size_t& num_edges() const { return _E; }
    
    // constructor
    Graph_P(const std::string& FileName, 
      size_t num_threads = std::thread::hardware_concurrency()) :
      _executor(num_threads),
      _pt_neighbors(num_threads),
      _pt_probabilities(num_threads),
      _pt_interblock_edge_count_r_row_new(num_threads),
      _pt_interblock_edge_count_s_row_new(num_threads),
      _pt_interblock_edge_count_r_col_new(num_threads),
      _pt_interblock_edge_count_s_col_new(num_threads),
      _pt_block_degrees_out_new(num_threads),
      _pt_block_degrees_in_new(num_threads),
      _pt_block_degrees_new(num_threads)  
    {
      load_graph_from_tsv(FileName);
      _generator.seed(_rd());
    }

    // partition ground truth
    std::vector<size_t> truePartitions;

  private:

    size_t _N; // number of node
    size_t _E; // number of edge
    
    std::vector<Edge> _edges;
    std::vector<std::vector<std::pair<size_t, W>>> _out_neighbors;
    std::vector<std::vector<std::pair<size_t, W>>> _in_neighbors;

    size_t _num_blocks;

    std::random_device _rd;
    std::default_random_engine _generator;

    // taskflow   
    tf::Executor _executor;
    tf::Taskflow _taskflow;
          
    std::vector< std::vector<size_t>> _pt_neighbors;
    std::vector< std::vector<float>> _pt_probabilities;
    std::vector< std::vector<W>> _pt_interblock_edge_count_r_row_new;
    std::vector< std::vector<W>> _pt_interblock_edge_count_s_row_new;
    std::vector< std::vector<W>> _pt_interblock_edge_count_r_col_new;
    std::vector< std::vector<W>> _pt_interblock_edge_count_s_col_new;
    std::vector< std::vector<W>> _pt_block_degrees_out_new;
    std::vector< std::vector<W>> _pt_block_degrees_in_new;
    std::vector< std::vector<W>> _pt_block_degrees_new;

    // save data for golden ratio bracket
    struct Old {
      std::vector<size_t> partitions_large;
      std::vector<size_t> partitions_med;
      std::vector<size_t> partitions_small;
      //std::vector< std::unordered_map<size_t, W> > Mrow_large;
      //std::vector< std::unordered_map<size_t, W> > Mrow_med;
      //std::vector< std::unordered_map<size_t, W> > Mrow_small;
      //std::vector< std::unordered_map<size_t, W> > Mcol_large;
      //std::vector< std::unordered_map<size_t, W> > Mcol_med;
      //std::vector< std::unordered_map<size_t, W> > Mcol_small;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_large2;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_med2;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_small2;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_large2;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_med2;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_small2;
      std::vector<W> block_degree_large;
      std::vector<W> block_degree_med;
      std::vector<W> block_degree_small;
      std::vector<W> block_degree_out_large;
      std::vector<W> block_degree_out_med;
      std::vector<W> block_degree_out_small;
      std::vector<W> block_degree_in_large;
      std::vector<W> block_degree_in_med;
      std::vector<W> block_degree_in_small;
      float overall_entropy_large;
      float overall_entropy_med;
      float overall_entropy_small;
      size_t num_blocks_large;
      size_t num_blocks_med;
      size_t num_blocks_small;
    };
    
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    //std::vector< std::vector<std::pair<size_t, W>> > Mrow;
    //std::vector< std::vector<std::pair<size_t, W>> > Mcol;
    ////////////////////////////////////////////////
    ////////////////////////////////////////////////


    // functions used internally
    void _initialize_edge_counts(
      const std::vector<size_t>& partitions,
      //std::vector< std::unordered_map<size_t, W> >& Mrow,
      //std::vector< std::unordered_map<size_t, W> >& Mcol,
      std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      std::vector<W>& d_out, 
      std::vector<W>& d_in, 
      std::vector<W>& d
    );
 
    void _propose_new_partition_block(
      size_t r,
      const std::vector<size_t>& partitions,
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      const std::vector<W>& d,
      const std::default_random_engine& generator,
      size_t& s,
      W& k_out,
      W& k_in,
      W& k,
      std::vector<size_t>& neighbors,
      std::vector<float>& prob
    );
 
    void _propose_new_partition_nodal(
      size_t r,
      size_t ni,
      const std::vector<size_t>& partitions,
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      const std::vector<W>& d,
      const std::default_random_engine& generator,
      size_t& s,
      W& k_out,
      W& k_in,
      W& k,
      std::vector<size_t>& neighbors,
      std::vector<float>& prob
    );

    void _compute_new_rows_cols_interblock_edge_count_block(
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      size_t r,
      size_t s,
      std::vector<W>& M_r_row,
      std::vector<W>& M_s_row,
      std::vector<W>& M_r_col,
      std::vector<W>& M_s_col
    );

    void _compute_new_rows_cols_interblock_edge_count_nodal(
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      size_t r,
      size_t s,
      size_t ni,
      const std::vector<size_t>& partitions,
      std::vector<W>& M_r_row,
      std::vector<W>& M_s_row,
      std::vector<W>& M_r_col,
      std::vector<W>& M_s_col
    );
    
    void _compute_new_block_degree(
      size_t r,
      size_t s,
      const std::vector<W>& d_out,
      const std::vector<W>& d_in,
      const std::vector<W>& d,
      W k_out,
      W k_in,
      W k,
      std::vector<W>& d_out_new,
      std::vector<W>& d_in_new,
      std::vector<W>& d_new
    );
                                  
    float _compute_delta_entropy(
      size_t r,
      size_t s,
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_s_row,
      const std::vector<W>& M_r_col,
      const std::vector<W>& M_s_col,
      const std::vector<W>& d_out,
      const std::vector<W>& d_in,
      const std::vector<W>& d_out_new,
      const std::vector<W>& d_in_new
    );        
     
    size_t _carry_out_best_merges(
      const std::vector<size_t>& bestMerges,
      const std::vector<int>& best_merge_for_each_block,
      size_t B_to_merge,
      std::vector<size_t>& b,
      std::vector<size_t>& block_map,
      std::vector<size_t>& remaining_blocks
    );

    float _compute_overall_entropy(
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      const std::vector<W>& d_out,
      const std::vector<W>& d_in
    ); 

    float _compute_Hastings_correction(
      size_t s,
      size_t ni,
      const std::vector<size_t>& partitions,
      //TODO: change the M to class 
      //const std::vector< std::unordered_map<size_t, W> >& Mrow,
      //const std::vector< std::unordered_map<size_t, W> >& Mcol,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_r_col,
      const std::vector<W>& d,
      const std::vector<W>& d_new
    );

    void _update_partition(
      size_t ni,
      size_t r,
      size_t s,
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_s_row,
      const std::vector<W>& M_r_col,
      const std::vector<W>& M_s_col,
      const std::vector<W>& d_out_new,
      const std::vector<W>& d_in_new,
      const std::vector<W>& d_new,
      std::vector<size_t>& b,
      //std::vector< std::unordered_map<size_t, W> >& Mrow,
      //std::vector< std::unordered_map<size_t, W> >& Mcol,
      std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      std::vector<W>& d_out,
      std::vector<W>& d_in,
      std::vector<W>& d
    );

    bool _prepare_for_partition_next(
      float S,
      std::vector<size_t>& b,
      //std::vector< std::unordered_map<size_t, W> >& Mrow,
      //std::vector< std::unordered_map<size_t, W> >& Mcol,
      std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
      std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
      std::vector<W>& d,
      std::vector<W>& d_out,
      std::vector<W>& d_in,
      size_t& B,
      size_t& B_to_merge,
      Old& old,
      float B_rate
    );

    // utility functions
    std::vector<size_t> _argsort(const std::vector<float>& arr) {
      std::vector<size_t> sorted_indices(arr.size());
      std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
      std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&arr](int i, int j){ return arr[i] < arr[j]; });
      return sorted_indices;
    }

    std::vector<size_t> _unique(const std::vector<size_t>& arr) {
      std::vector<size_t> unique_arr;
      std::set<size_t> seen;
      for (auto elem : arr) {
        if (seen.find(elem) == seen.end()) {
          unique_arr.push_back(elem);
          seen.insert(elem);
        }   
      }
      std::sort(unique_arr.begin(), unique_arr.end());
      return unique_arr;
    }

    bool _compare_if_vec(std::vector<W> v1, std::vector<W> v2) {
      if (v1.size() != v2.size()) {
        return false;
      } 
      else {
        for (size_t i = 0; i < v1.size(); i++) {
          if (v1[i] != v2[i]) {
            return false;
          }
        }
      }
      return true;
    }

    bool _compare_if_M(std::vector< std::unordered_map<size_t, W> > M1, std::vector< std::vector<std::pair<size_t, W>> > M2) {
      if (M1.size() != M2.size()) {
        return false;
      }
      std::vector< std::unordered_map<size_t, W> > test;
      test.resize(M1.size());
      for (size_t i = 0; i < test.size(); i++) {
        for (const auto& [v, w] : M2[i]) {
          test[i][v] += w;
        }
      }
      for (size_t i = 0; i < test.size(); i++) {
        for (const auto& [v, w] : test[i]) {
          if ( w != M1[i][v] ) {
            return false;
          }
        }
      }
      return true;
    }
  

}; // end of class Graph_P


// function definitions
//
//
template <typename W>
void Graph_P<W>::load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv");
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }

  _N = 0;

  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;
  size_t from, to;
  W weight;
  while (std::getline(file, line)) {
    size_t start = 0;
    size_t tab_pos = line.find('\t');
    from = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    to = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    weight = static_cast<W>(std::stof(line.substr(start, tab_pos - start)));
    _edges.emplace_back(Edge {from, to, weight});
    if (from > _N) _N = from;
  }
  file.close();

  _E = _edges.size();
  
  _out_neighbors.resize(_N);
  _in_neighbors.resize(_N);
  
  for (const auto& e : _edges) {
    _out_neighbors[e.from-1].emplace_back(e.to-1, e.weight);
    _in_neighbors[e.to-1].emplace_back(e.from-1, e.weight);
  }

  // load the true partition
  std::ifstream true_file(FileName + "_truePartition.tsv");
  if (!true_file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  // format: node i \t block
  while (std::getline(true_file, line)) {
    size_t start = 0;
    size_t tab_pos = line.find('\t');
    size_t i = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    size_t block = std::stoi(line.substr(start, tab_pos - start));
    truePartitions.emplace_back(block-1);
  }
  true_file.close();
} // end of load_graph_from_tsv

template <typename W>
std::vector<size_t> Graph_P<W>::partition() {
  
  _num_blocks = _N;
  std::vector<size_t> partitions(_num_blocks);
  std::iota(partitions.begin(), partitions.end(), 0); // initialize the partition
  
  //std::vector< std::unordered_map<size_t, W> > Mrow;
  //std::vector< std::unordered_map<size_t, W> > Mcol;

  std::vector< std::vector<std::pair<size_t, W>> > Mrow2;
  std::vector< std::vector<std::pair<size_t, W>> > Mcol2;
  //Mrow.clear();
  //Mcol.clear();

  std::vector<W> block_degrees_out;
  std::vector<W> block_degrees_in;
  std::vector<W> block_degrees;

  _initialize_edge_counts(
    partitions, 
    //Mrow,
    //Mcol,
    Mrow2,
    Mcol2,
    block_degrees_out, 
    block_degrees_in, 
    block_degrees
  );

  Old _old;
  
  _old.overall_entropy_large = std::numeric_limits<float>::infinity();
  _old.overall_entropy_med = std::numeric_limits<float>::infinity();
  _old.overall_entropy_small = std::numeric_limits<float>::infinity();

  _old.num_blocks_large = 0;
  _old.num_blocks_med = 0;
  _old.num_blocks_small = 0;

  bool optimal_num_blocks_found = false;

  size_t num_blocks_to_merge = (size_t)_num_blocks * num_block_reduction_rate;

  std::vector<int> best_merge_for_each_block;
  std::vector<float> delta_entropy_for_each_block;
  std::vector<size_t> bestMerges;
  std::vector<size_t> block_map;
  std::vector<size_t> block_partition;

  // proposal
  std::vector<size_t> neighbors_nodal;
  std::vector<float> prob_nodal;

  std::vector<W> interblock_edge_count_r_row_new;
  std::vector<W> interblock_edge_count_s_row_new;
  std::vector<W> interblock_edge_count_r_col_new;
  std::vector<W> interblock_edge_count_s_col_new;

  std::vector<W> block_degrees_out_new;
  std::vector<W> block_degrees_in_new;
  std::vector<W> block_degrees_new;
  
  // for merge
  std::vector<size_t> remaining_blocks;

  std::vector<float> itr_delta_entropy;

  // timing
  //auto block_merge_time = 0;
  //auto nodal_update_time = 0;

  while (!optimal_num_blocks_found) {
    if (verbose)  
      printf("\nMerging down blocks from %ld to %ld\n", 
              _num_blocks, 
              _num_blocks - num_blocks_to_merge
            );

    // init record for the round
    best_merge_for_each_block.clear();
    best_merge_for_each_block.resize(_num_blocks, -1);
    delta_entropy_for_each_block.clear();
    delta_entropy_for_each_block.resize(_num_blocks, std::numeric_limits<float>::infinity());
    block_partition.clear();
    block_partition.resize(_num_blocks, 0);
    std::iota(block_partition.begin(), block_partition.end(), 0);

    // block merge
    //auto block_merge_start = std::chrono::steady_clock::now();
    
    // TODO: try to use for_each instead of explicitly creating B tasks
    _taskflow.clear();
    for (size_t current_block = 0; current_block < _num_blocks; current_block++) {
      _taskflow.emplace([this,
        //&Mrow,
        //&Mcol,
        &Mrow2,
        &Mcol2,
        &block_partition,
        &block_degrees,
        &block_degrees_in,
        &block_degrees_out,
        &best_merge_for_each_block,
        &delta_entropy_for_each_block,
        current_block ](){
          
          auto wid = _executor.this_worker_id();
          auto& neighbors = _pt_neighbors[wid];
          auto& prob = _pt_probabilities[wid];
          auto& interblock_edge_count_r_row_new = _pt_interblock_edge_count_r_row_new[wid];
          auto& interblock_edge_count_s_row_new = _pt_interblock_edge_count_s_row_new[wid];
          auto& interblock_edge_count_r_col_new = _pt_interblock_edge_count_r_col_new[wid];
          auto& interblock_edge_count_s_col_new = _pt_interblock_edge_count_s_col_new[wid];
          auto& block_degrees_out_new = _pt_block_degrees_out_new[wid];
          auto& block_degrees_in_new = _pt_block_degrees_in_new[wid];
          auto& block_degrees_new = _pt_block_degrees_new[wid];

          for (size_t proposal_idx = 0; proposal_idx < num_agg_proposals_per_block; proposal_idx++) {
            
            neighbors.clear();
            prob.clear();
            interblock_edge_count_r_row_new.clear();
            interblock_edge_count_s_row_new.clear();
            interblock_edge_count_r_col_new.clear();
            interblock_edge_count_s_col_new.clear();
            block_degrees_out_new.clear();
            block_degrees_in_new.clear();
            block_degrees_new.clear();

            size_t proposal;
            W num_out_neighbor_edges;
            W num_in_neighbor_edges;
            W num_neighbor_edges;
            _propose_new_partition_block(
              current_block,
              block_partition,
              //Mrow,
              //Mcol,
              Mrow2,
              Mcol2,
              block_degrees,
              _generator,
              proposal,
              num_out_neighbor_edges,
              num_in_neighbor_edges,
              num_neighbor_edges,
              neighbors,
              prob
            );   

            interblock_edge_count_r_row_new.clear();
            interblock_edge_count_s_row_new.clear();
            interblock_edge_count_r_col_new.clear();
            interblock_edge_count_s_col_new.clear();
            _compute_new_rows_cols_interblock_edge_count_block(
              //Mrow,
              //Mcol,
              Mrow2,
              Mcol2,
              current_block,
              proposal,
              interblock_edge_count_r_row_new,
              interblock_edge_count_s_row_new,
              interblock_edge_count_r_col_new,
              interblock_edge_count_s_col_new
            );

            block_degrees_out_new.clear();
            block_degrees_in_new.clear();
            block_degrees_new.clear();
            _compute_new_block_degree(
              current_block,
              proposal,
              block_degrees_out,
              block_degrees_in,
              block_degrees,
              num_out_neighbor_edges,
              num_in_neighbor_edges,
              num_neighbor_edges,
              block_degrees_out_new,
              block_degrees_in_new,
              block_degrees_new
            );
            
            float delta_entropy = _compute_delta_entropy(
              current_block,
              proposal,
              //Mrow,
              //Mcol,
              Mrow2,
              Mcol2,
              interblock_edge_count_r_row_new,
              interblock_edge_count_s_row_new,
              interblock_edge_count_r_col_new,
              interblock_edge_count_s_col_new,
              block_degrees_out,
              block_degrees_in,
              block_degrees_out_new,
              block_degrees_in_new
            );
            
            if (delta_entropy < delta_entropy_for_each_block[current_block]) {
              best_merge_for_each_block[current_block] = proposal;
              delta_entropy_for_each_block[current_block] = delta_entropy;
            }     
          } // end for proposal_idx
      });
    }
    
    _executor.run(_taskflow).wait();
    //auto block_merge_end = std::chrono::steady_clock::now();
    //block_merge_time += std::chrono::duration_cast<std::chrono::milliseconds>
    //                      (block_merge_end - block_merge_start).count();


    bestMerges = _argsort(delta_entropy_for_each_block);
    _num_blocks = _carry_out_best_merges(
                    bestMerges,
                    best_merge_for_each_block,
                    num_blocks_to_merge,
                    partitions,
                    block_map,
                    remaining_blocks
                  );

    //Mrow.clear();
    //Mcol.clear();
    Mrow2.clear();
    Mcol2.clear();
    block_degrees_out.clear();
    block_degrees_in.clear();
    block_degrees.clear();
    _initialize_edge_counts(
      partitions, 
      //Mrow,
      //Mcol,
      Mrow2,
      Mcol2,
      block_degrees_out, 
      block_degrees_in, 
      block_degrees
    );

    int total_num_nodal_moves = 0;
    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);

    float overall_entropy = _compute_overall_entropy(
      //Mrow,
      //Mcol,
      Mrow2,
      Mcol2,
      block_degrees_out, 
      block_degrees_in
    );
    
    if (verbose)
      printf("overall_entropy: %f\n", overall_entropy);
 
    // nodal updates
    //auto nodal_update_start = std::chrono::steady_clock::now();
    for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {

      int num_nodal_moves = 0;
      itr_delta_entropy[itr] = 0;

      for (size_t current_node = 0; current_node < _N; current_node++) {
      
        size_t current_block = partitions[current_node];
     
        neighbors_nodal.clear();
        prob_nodal.clear(); 
        size_t proposal;
        W num_out_neighbor_edges;
        W num_in_neighbor_edges;
        W num_neighbor_edges;
        _propose_new_partition_nodal(
          current_block,
          current_node,
          partitions,
          //Mrow,
          //Mcol,
          Mrow2,
          Mcol2,
          block_degrees,
          _generator,
          proposal,
          num_out_neighbor_edges,
          num_in_neighbor_edges,
          num_neighbor_edges,
          neighbors_nodal,
          prob_nodal
        );
        
        if (proposal != current_block) {
          
          interblock_edge_count_r_row_new.clear();
          interblock_edge_count_s_row_new.clear();
          interblock_edge_count_r_col_new.clear();
          interblock_edge_count_s_col_new.clear();
          _compute_new_rows_cols_interblock_edge_count_nodal(
            //Mrow,
            //Mcol,
            Mrow2,
            Mcol2,
            current_block,
            proposal,
            current_node,
            partitions,
            interblock_edge_count_r_row_new,
            interblock_edge_count_s_row_new,
            interblock_edge_count_r_col_new,
            interblock_edge_count_s_col_new
          );
          
          block_degrees_out_new.clear();
          block_degrees_in_new.clear();
          block_degrees_new.clear();
          _compute_new_block_degree(
            current_block,
            proposal,
            block_degrees_out,
            block_degrees_in,
            block_degrees,
            num_out_neighbor_edges,
            num_in_neighbor_edges,
            num_neighbor_edges,
            block_degrees_out_new,
            block_degrees_in_new,
            block_degrees_new
          );

          float Hastings_correction = 1.0;
          if (num_neighbor_edges > 0) {
            Hastings_correction = _compute_Hastings_correction(
              proposal,
              current_node,
              partitions,
              //Mrow,
              //Mcol,
              Mrow2,
              Mcol2,
              interblock_edge_count_r_row_new,
              interblock_edge_count_r_col_new,
              block_degrees,
              block_degrees_new
            );
          } // calculate Hastings_correction
          
          float delta_entropy = _compute_delta_entropy(
            current_block,
            proposal,
            //Mrow,
            //Mcol,
            Mrow2,
            Mcol2,
            interblock_edge_count_r_row_new,
            interblock_edge_count_s_row_new,
            interblock_edge_count_r_col_new,
            interblock_edge_count_s_col_new,
            block_degrees_out,
            block_degrees_in,
            block_degrees_out_new,
            block_degrees_in_new
          );

          float p_accept = std::min(
            static_cast<float>(std::exp(-beta * delta_entropy)) * Hastings_correction, 
            1.0f
          );


          std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
          float rand_num = uni_dist(_generator);
          if ( rand_num <= p_accept) {
            total_num_nodal_moves++;
            num_nodal_moves++;
            itr_delta_entropy[itr] += delta_entropy;
            _update_partition(
              current_node,
              current_block,
              proposal,
              interblock_edge_count_r_row_new,
              interblock_edge_count_s_row_new,
              interblock_edge_count_r_col_new,
              interblock_edge_count_s_col_new,
              block_degrees_out_new,
              block_degrees_in_new,
              block_degrees_new,
              partitions,
              //Mrow,
              //Mcol,
              Mrow2,
              Mcol2,
              block_degrees_out,
              block_degrees_in,
              block_degrees
            );
          }
        } // end if 
      } // end current_node

      float oe = overall_entropy = _compute_overall_entropy(
        //Mrow,
        //Mcol,
        Mrow2,
        Mcol2,
        block_degrees_out, 
        block_degrees_in
      );

      if (verbose)
        printf("Itr: %ld, number of nodal moves: %d, delta S: %.5f, overall_entropy:%f \n", 
                itr, num_nodal_moves, itr_delta_entropy[itr] / float(overall_entropy), oe);
      
      if (itr >= (delta_entropy_moving_avg_window - 1)) {
        bool isfinite = true;
        isfinite = isfinite && std::isfinite(_old.overall_entropy_large);
        isfinite = isfinite && std::isfinite(_old.overall_entropy_med);
        isfinite = isfinite && std::isfinite(_old.overall_entropy_small);
        float mean = 0;
        for (int i = itr - delta_entropy_moving_avg_window + 1; i < itr; i++) {
          mean += itr_delta_entropy[i];
        }   
        mean /= (float)(delta_entropy_moving_avg_window - 1); 
        if (!isfinite) {
          if (-mean < (delta_entropy_threshold1 * overall_entropy)) {
            if (verbose)
              printf("golden ratio bracket is not yet established: %.5f %.5f\n", 
                      -mean, delta_entropy_threshold1 * overall_entropy);
            break;
          }   
        }   
        else {
          if (-mean < (delta_entropy_threshold2 * overall_entropy)) {
            if (verbose)
              printf("golden ratio bracket is established: %f \n", -mean);
            break;
          }   
        }   
      }   
    } // end itr
    //auto nodal_update_end = std::chrono::steady_clock::now();
    //nodal_update_time += std::chrono::duration_cast<std::chrono::milliseconds>
    //                      (nodal_update_end - nodal_update_start).count();
  //std::cout << "block_merge_time: " << block_merge_time << std::endl;
  //std::cout << "block_merge_propose_time: " << block_merge_propose_time << std::endl;
  //std::cout << "block_merge_compute_new_time: " << block_merge_compute_new_time << std::endl;
  //std::cout << "block_merge_compute_entropy_time: " << block_merge_compute_entropy_time << std::endl;
  //std::cout << "nodal_update_time: " << nodal_update_time << std::endl;
  //std::cout << "nodal_update_propose_time: " << nodal_update_propose_time << std::endl;
  //std::cout << "nodal_update_compute_new_time: " << nodal_update_compute_new_time << std::endl;
  //std::cout << "nodal_update_compute_entropy_time: " << nodal_update_compute_entropy_time << std::endl;
    overall_entropy = _compute_overall_entropy(
      //Mrow,
      //Mcol,
      Mrow2,
      Mcol2,
      block_degrees_out, 
      block_degrees_in
    );

    if (verbose)
      printf("Total number of nodal moves: %d, overall_entropy: %.5f\n", 
              total_num_nodal_moves, overall_entropy);


    optimal_num_blocks_found = _prepare_for_partition_next(
      overall_entropy,
      partitions,
      //Mrow,
      //Mcol,
      Mrow2,
      Mcol2,
      block_degrees,
      block_degrees_out,
      block_degrees_in,
      _num_blocks, //TODO
      num_blocks_to_merge,
      _old,
      num_block_reduction_rate
    );

    if (verbose) {
      printf("Overall entropy: [%f, %f, %f] \n", 
        _old.overall_entropy_large, _old.overall_entropy_med, _old.overall_entropy_small);
      printf("Number of blocks: [%ld, %ld, %ld] \n",
        _old.num_blocks_large, _old.num_blocks_med, _old.num_blocks_small);
    }

  } // end while
  //std::cout << "block_merge_time: " << block_merge_time << std::endl;
  //std::cout << "nodal_update_time: " << nodal_update_time << std::endl;

  return partitions;
}

template <typename W>
void Graph_P<W>::_initialize_edge_counts(
  const std::vector<size_t>& partitions,
  //std::vector< std::unordered_map<size_t, W> >& Mrow,
  //std::vector< std::unordered_map<size_t, W> >& Mcol, 
  std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  std::vector<W>& d_out, 
  std::vector<W>& d_in, 
  std::vector<W>& d) 
{

  //_taskflow.clear();
  Mrow.clear();
  Mcol.clear();
  Mrow.resize(_num_blocks);
  Mcol.resize(_num_blocks);
 
  
  //Mrow2.clear();
  //Mcol2.clear();
  //Mrow2.resize(_num_blocks);
  //Mcol2.resize(_num_blocks); 
  
  for (size_t node = 0; node < _out_neighbors.size(); node++) {
    if (_out_neighbors[node].size() > 0) {
      size_t k1 = partitions[node];
      for (const auto& [v, w] : _out_neighbors[node]) {
        size_t k2 = partitions[v];
        //Mrow[k1][k2] += w;
        //Mcol[k2][k1] += w;
        //_Mrow2[k1].emplace_back(std::make_pair(k2, w));
        //_Mcol2[k2].emplace_back(std::make_pair(k1, w));
        Mrow[k1].emplace_back(std::make_pair(k2, w));
        Mcol[k2].emplace_back(std::make_pair(k1, w));
      }
    }
  }

  /*
  if (!_compare_if_M(Mrow, Mrow2)) {
    std::cout << "init Mrow diff\n";
    std::exit(1);
  }
  if (!_compare_if_M(Mcol, Mcol2)) {
    std::cout << "init Mcol diff\n";
    std::exit(1);
  }
  */

  d_out.clear();
  d_out.resize(_num_blocks);
  d_in.clear();
  d_in.resize(_num_blocks);
  d.clear();
  d.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow[i]) {
      d_out[i] += w;
    }
    for (const auto& [v, w] : Mcol[i]) {
      d_in[i] += w;
    }
    d[i] = d_out[i] + d_in[i];
  }

  /////////////////////////////////////////////
  ////////////////////////////////////////////
  /*
  std::vector<W> d_out2;
  d_out2.resize(_num_blocks);
  std::vector<W> d_in2;
  d_in2.resize(_num_blocks);
  std::vector<W> d2;
  d2.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow2[i]) {
      d_out2[i] += w;
    }    
    for (const auto& [v, w] : Mcol2[i]) {
      d_in2[i] += w;
    }    
    d2[i] = d_out2[i] + d_in2[i];
  }
  if (!_compare_if_vec(d_out, d_out2)) {
    std::cout << "init d_out diff\n";
    std::exit(1);
  }
  if (!_compare_if_vec(d_in, d_in2)) {
    std::cout << "init d_in diff\n";
    std::exit(1);
  }
  if (!_compare_if_vec(d, d2)) {
    std::cout << "init d diff\n";
    std::exit(1);
  }
  */
  /////////////////////////////////////////////
  /////////////////////////////////////////////

} // end of initialize_edge_counts

template <typename W>
void Graph_P<W>::_propose_new_partition_block(
  size_t r,
  const std::vector<size_t>& partitions,
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  const std::vector<W>& d,
  const std::default_random_engine& generator,
  size_t& s,
  W& k_out,
  W& k_in,
  W& k,
  std::vector<size_t>& neighbors,
  std::vector<float>& prob)
{
  ////////////////////////////////////
  //////////////////////////////////
  //if (!_compare_if_M(Mrow, Mrow2)) {
  //  std::cout << "propose block Mrow diff\n";
  //  std::exit(1);
  //}
  //if (!_compare_if_M(Mcol, Mcol2)) {
  //  std::cout << "propose block  Mcol diff\n";
  //  std::exit(1);
  //}
  //////////////////////////////////
  //////////////////////////////


  k_out = 0;
  k_in = 0;
  //for (const auto& [v, w] : Mrow[r]) {
  //  k_out += w;
  //  neighbors.emplace_back(v);
  //  prob.emplace_back((float)w);
  //}
  //for (const auto& [v, w] : Mcol[r]) {
  //  k_in += w;
  //  neighbors.emplace_back(v);
  //  prob.emplace_back((float)w);
  //}
  
  prob.resize(_num_blocks);
  for (const auto& [v, w] : Mrow[r]) {
    k_out += w;
    prob[v] += w;
  }
  for (const auto& [v, w] : Mcol[r]) {
    k_in += w;
    prob[v] += w;
  }
  
  k = k_out + k_in;
  
  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if ( k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(), 
    [k](float p){ return p/(float)k; }
  );
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  
  //TODO: do i need neighbor??
  //size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(generator))];
  size_t rand_n = dist(const_cast<std::default_random_engine&>(generator));

  size_t u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(generator));
  if ( x <= (float)_num_blocks/(d[u]+_num_blocks) ) {
    //neighbors.clear();  
    //neighbors.resize(_num_blocks); 
    //std::iota(neighbors.begin(), neighbors.end(), 0);
    //std::uniform_int_distribution<int> choice(0, neighbors.size()-1);
    std::uniform_int_distribution<int> choice(0, _num_blocks-1);
    int randIndex = choice(const_cast<std::default_random_engine&>(generator));
    if (randIndex == r) randIndex++;
    if (randIndex == _num_blocks) randIndex = 0;
    //s = neighbors[randIndex];
    s = randIndex;
  }
  else {
    prob.clear();
    prob.resize(_num_blocks);
    float multinomial_prob_sum = 0;
    for (const auto& [v, w] : Mrow[u]) {
      prob[v] += w;
    }
    for (const auto& [v, w] : Mcol[u]) {
      prob[v] += w;
    }
    /*
     *
int sum = std::reduce(myMap.begin(), myMap.end(), 0,
                        [](int acc, const auto& elem) {
                          return acc + elem.second;
                        });
    */
    
    std::transform(prob.begin(), prob.end(), prob.begin(), 
      [&d, u](float p){ 
        return p/(float)d[u]; 
      }
    );
    multinomial_prob_sum = std::reduce(prob.begin(), prob.end(), 0.0);
    multinomial_prob_sum -= prob[r];
    prob[r] = 0;
    if (multinomial_prob_sum == 0) {
      //neighbors.clear();
      //neighbors.resize(_num_blocks);
      //std::iota(neighbors.begin(), neighbors.end(), 0);
      //std::uniform_int_distribution<int> choice(0, neighbors.size()-1);
      std::uniform_int_distribution<int> choice(0, _num_blocks-1);
      int randIndex = choice(const_cast<std::default_random_engine&>(generator));
      if (randIndex == r) randIndex++;
      if (randIndex == _num_blocks) randIndex = 0; 
      //s = neighbors[randIndex];
      s = randIndex;
      return;
    }
    else {
      std::transform(prob.begin(), prob.end(), prob.begin(), 
        [multinomial_prob_sum](float p){ 
          return p/multinomial_prob_sum;
        }
      );
    }
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }
}


template <typename W>
void Graph_P<W>::_propose_new_partition_nodal(
  size_t r,
  size_t ni,
  const std::vector<size_t>& partitions,
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  const std::vector<W>& d,
  const std::default_random_engine& generator,
  size_t& s,
  W& k_out,
  W& k_in,
  W& k,
  std::vector<size_t>& neighbors,
  std::vector<float>& prob) 
{

  neighbors.clear();
  prob.clear();
  k_out = 0;
  k_in = 0;

  for (const auto& [v, w] : _out_neighbors[ni]) {
    neighbors.emplace_back(v);
    prob.emplace_back((float)w);
    k_out += w;
  }
  for (const auto& [v, w] : _in_neighbors[ni]) {
    neighbors.emplace_back(v);
    prob.emplace_back((float)w);
    k_in += w;
  }
   
  
  k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if (k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(), 
    [k](float p){
      return p/(float)k;
    }
  );
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(generator))];
  size_t u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(generator));
  if ( x <= (float)_num_blocks/(d[u]+_num_blocks) ) {
    s = randint(const_cast<std::default_random_engine&>(generator));
  }
  else {
    prob.clear();
    prob.resize(_num_blocks);
    for (const auto& [v, w] : Mrow[u]) {
      prob[v] += w;
    }    
    for (const auto& [v, w] : Mcol[u]) {
      prob[v] += w;
    }    
    std::transform(prob.begin(), prob.end(), prob.begin(),
      [&d, u](float p){
        return p/(float)d[u];
      }
    );
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }
}

template <typename W>
void Graph_P<W>::_compute_new_rows_cols_interblock_edge_count_block(
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol, 
  size_t r,
  size_t s,
  std::vector<W>& M_r_row,
  std::vector<W>& M_s_row,
  std::vector<W>& M_r_col,
  std::vector<W>& M_s_col)
{
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks);
  W count_in_sum_s = 0;
  W count_out_sum_s = 0;
  W count_self = 0;
  for (const auto& [v, w] : Mrow[s]) {
    M_s_row[v] += w;
  }
  for (const auto& [v, w] : Mcol[s]) {
    M_s_col[v] += w;
  }
  for (const auto& [v, w] : Mrow[r]) {
    if (v == s) {
      count_out_sum_s += w;
    }
    if (v == r) {
      count_self += w;
    }
    M_s_row[v] += w;
  }
  for (const auto& [v, w] : Mcol[r]) {
    if (v == s) {
      count_in_sum_s += w;
    }
    M_s_col[v] += w;
  }
  M_s_row[r] -= count_in_sum_s;
  M_s_row[s] += count_in_sum_s;
  M_s_row[r] -= count_self;
  M_s_row[s] += count_self;
  M_s_col[r] -= count_out_sum_s;
  M_s_col[s] += count_out_sum_s;
  M_s_col[r] -= count_self;
  M_s_col[s] += count_self;
}


template <typename W>
void Graph_P<W>::_compute_new_rows_cols_interblock_edge_count_nodal(
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  size_t r,
  size_t s,
  size_t ni,
  const std::vector<size_t>& partitions,
  std::vector<W>& M_r_row,
  std::vector<W>& M_s_row,
  std::vector<W>& M_r_col,
  std::vector<W>& M_s_col) 
{
 
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks); 
  W count_out_sum_r = 0;
  W count_in_sum_r = 0;
  W count_out_sum_s = 0;
  W count_in_sum_s = 0;
  W count_self = 0;
  for (const auto& [v, w] : Mrow[r]) {
    M_r_row[v] += w;
  }
  for (const auto& [v, w] : Mcol[r]) {
    M_r_col[v] += w;
  }
  for (const auto& [v, w] : Mrow[s]) {
    M_s_row[v] += w;
  }
  for (const auto& [v, w] : Mcol[s]) {
    M_s_col[v] += w;
  }
  
  size_t b;
  for (const auto& [v, w] : _out_neighbors[ni]) {
    b = partitions[v];
    if (b == r) {
      count_out_sum_r += w;
    }
    if (b == s) {
      count_out_sum_s += w;
    }
    if (v == ni) {
      count_self += w;
    }
    M_r_row[b] -= w;
    M_s_row[b] += w;
  }
  for (const auto& [v, w] : _in_neighbors[ni]) {
    b = partitions[v];
    if (b == r) {
      count_in_sum_r += w;
    }
    if (b == s) {
      count_in_sum_s += w;
    }
    M_r_col[b] -= w;
    M_s_col[b] += w;
  }
  
  M_r_row[r] -= count_in_sum_r;
  M_r_row[s] += count_in_sum_r;
  M_r_col[r] -= count_out_sum_r;
  M_r_col[s] += count_out_sum_r;

  M_s_row[r] -= count_in_sum_s;
  M_s_row[s] += count_in_sum_s;
  M_s_row[r] -= count_self;
  M_s_row[s] += count_self;
  M_s_col[r] -= count_out_sum_s;
  M_s_col[s] += count_out_sum_s;
  M_s_col[r] -= count_self;
  M_s_col[s] += count_self;

} // end of compute_new_rows_cols_interblock_edge_count_matrix

template <typename W>
void Graph_P<W>::_compute_new_block_degree(
  size_t r,
  size_t s,
  const std::vector<W>& d_out,
  const std::vector<W>& d_in,
  const std::vector<W>& d,
  W k_out,
  W k_in,
  W k,
  std::vector<W>& d_out_new,
  std::vector<W>& d_in_new,
  std::vector<W>& d_new) 
{

  d_out_new = d_out;
  d_in_new = d_in;
  d_new = d;

  d_out_new[r] -= k_out;
  d_out_new[s] += k_out;

  d_in_new[r] -= k_in;
  d_in_new[s] += k_in;

  d_new[r] -= k;
  d_new[s] += k;

} // end of compute_new_block_degree


template <typename W>
float Graph_P<W>::_compute_delta_entropy(
  size_t r,
  size_t s,
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_s_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& M_s_col,
  const std::vector<W>& d_out,
  const std::vector<W>& d_in,
  const std::vector<W>& d_out_new,
  const std::vector<W>& d_in_new) 
{
  /*
  float delta_entropy = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_row[i] != 0) {
      delta_entropy -= M_r_row[i] * std::log(static_cast<float>
                                      (M_r_row[i]) / (d_in_new[i] * d_out_new[r])
                                    );
    }
    if (M_s_row[i] != 0) {
      delta_entropy -= M_s_row[i] * std::log(static_cast<float>
                                      (M_s_row[i]) / (d_in_new[i] * d_out_new[s])
                                    );
    }
    // avoid duplicate counting
    if (i != r && i != s) {
      if (M_r_col[i] != 0) {
        delta_entropy -= M_r_col[i] * std::log(static_cast<float>
                                        (M_r_col[i]) / (d_out_new[i] * d_in_new[r])
                                      );
      }
      if (M_s_col[i] != 0) {
        delta_entropy -= M_s_col[i] * std::log(static_cast<float>
                                        (M_s_col[i]) / (d_out_new[i] * d_in_new[s])
                                      );
      }
    }
  }
  //////////////////////
  /////////////////////
  float Mrowr = 0;
  float Mrows = 0;
  float Mcolr = 0;
  float Mcols = 0;
  ///////////////////
  //////////////////


  for (const auto& [v, w] : Mrow[r]) {
    if (w!=0) { //TODO:?
      delta_entropy += w * std::log(static_cast<float> (w) / (d_in[v] * d_out[r]));
      Mrowr += w * std::log(static_cast<float> (w) / (d_in[v] * d_out[r]));
    }
  }
  for (const auto& [v, w] : Mrow[s]) {
    if (w!=0) {
      delta_entropy += w * std::log(static_cast<float> (w) / (d_in[v] * d_out[s]));
      Mrows += w * std::log(static_cast<float> (w) / (d_in[v] * d_out[s]));
    }
  }
  for (const auto& [v, w] : Mcol[r]) {
    if (v != r && v != s) {
      if (w != 0) {
        delta_entropy += w * std::log(static_cast<float> (w) / (d_out[v] * d_in[r]));
        Mcolr += w * std::log(static_cast<float> (w) / (d_out[v] * d_in[r]));
      }
    }
  }
  for (const auto& [v, w] : Mcol[s]) {
    if (v != r && v != s) {
      if (w != 0) {
        delta_entropy += w * std::log(static_cast<float> (w) / (d_out[v] * d_in[s]));
        Mcols += w * std::log(static_cast<float> (w) / (d_out[v] * d_in[s]));
      }
    }
  }
  */
  ////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  ////////////////////////////////////
  //////////////////////////////////
  /*
  std::vector< std::unordered_map<size_t, W> > f(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow2[i]) {
      f[i][v] += w;
    }
  }
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow[i]) {
      if (w != f[i][v]) {
        std::cout << "Mrow check error\n";
        std::exit(1);
      }
    }    
  } 
  */
  
  
  //////////////////////////////////
  //////////////////////////////
  
  float delta_entropy2 = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_row[i] != 0) {
      delta_entropy2 -= M_r_row[i] * std::log(static_cast<float>
                                      (M_r_row[i]) / (d_in_new[i] * d_out_new[r])
                                    );
    }
    if (M_s_row[i] != 0) {
      delta_entropy2 -= M_s_row[i] * std::log(static_cast<float>
                                      (M_s_row[i]) / (d_in_new[i] * d_out_new[s])
                                    );
    }
    // avoid duplicate counting
    if (i != r && i != s) {
      if (M_r_col[i] != 0) {
        delta_entropy2 -= M_r_col[i] * std::log(static_cast<float>
                                        (M_r_col[i]) / (d_out_new[i] * d_in_new[r])
                                      );
      }
      if (M_s_col[i] != 0) {
        delta_entropy2 -= M_s_col[i] * std::log(static_cast<float>
                                        (M_s_col[i]) / (d_out_new[i] * d_in_new[s])
                                      );
      }
    }
  }
  // TODO if work fuse it
  //

  float Mrowr2 = 0; 
  float Mrows2 = 0; 
  float Mcolr2 = 0; 
  float Mcols2 = 0;

  std::vector<W> tmp(_num_blocks);
  std::vector<W> tmp1(_num_blocks);
  std::vector<W> tmp2(_num_blocks);
  std::vector<W> tmp3(_num_blocks);
  for (const auto& [v, w] : Mrow[r]) {
    if (w != 0)
      tmp[v] += w;
  }
  for (size_t v = 0; v < _num_blocks; v++) {
    if (tmp[v] != 0) { 
      delta_entropy2 += tmp[v] * std::log(static_cast<float> (tmp[v]) / (d_in[v] * d_out[r]));
      Mrowr2 += tmp[v] * std::log(static_cast<float> (tmp[v]) / (d_in[v] * d_out[r]));
    }
  }

  for (const auto& [v, w] : Mrow[s]) {
    if (w != 0) 
      tmp1[v] += w;
  }
  for (size_t v = 0; v < _num_blocks; v++) {
    if (tmp1[v] != 0) {
      delta_entropy2 += tmp1[v] * std::log(static_cast<float> (tmp1[v]) / (d_in[v] * d_out[s]));
      Mrows2 += tmp1[v] * std::log(static_cast<float> (tmp1[v]) / (d_in[v] * d_out[s]));
    }
  }
 
  for (const auto& [v, w] : Mcol[r]) {
    if (w != 0)
      tmp2[v] += w;
  }
  for (size_t v = 0; v < _num_blocks; v++) {
    if (v != r && v != s) {
      if (tmp2[v] != 0) {
        delta_entropy2 += tmp2[v] * std::log(static_cast<float> (tmp2[v]) / (d_out[v] * d_in[r]));
        Mcolr2 += tmp2[v] * std::log(static_cast<float> (tmp2[v]) / (d_out[v] * d_in[r]));
      }
    }
  }
 
  for (const auto& [v, w] : Mcol[s]) {
    if (w != 0)
      tmp3[v] += w;
  }
  for (size_t v = 0; v < _num_blocks; v++) {
    if (v != r && v != s) {
      if (tmp3[v] != 0) {
        delta_entropy2 += tmp3[v] * std::log(static_cast<float> (tmp3[v]) / (d_out[v] * d_in[s]));
        Mcols2 += tmp3[v] * std::log(static_cast<float> (tmp3[v]) / (d_out[v] * d_in[s]));
      }
    }
  }
 
  /* 
  if (std::abs(delta_entropy - delta_entropy2) > 5e-2) {
    std::cout << "delta_entropy error\n";
    std::cout << delta_entropy << " " << delta_entropy2 << std::endl;
    std::cout << Mrowr << " " << Mrowr2 << std::endl;
    for (const auto& [v, w] : Mrow[r]) {
      if (w != 0) {
        if (w != tmp[v]) {
          std::cout << "(" << v << "," << w << ") ";
          std::cout << "(" << v << "," << tmp[v] << ") ";
        }
      }
    }
    std::unordered_map<size_t, W> fuck;
    for (const auto& [v, w] : Mrow2[r]){
      fuck[v] += w;
    }
    std::cout << "fuck\n";
    for (const auto& [v, w] : Mrow[r]) {
      if (w != 0) {
        if (w != fuck[v]) {
          std::cout << "(" << v << "," << w << ") ";
          std::cout << "(" << v << "," << fuck[v] << ") ";
        }
      }
    }
    std::cout << std::endl;
    std::cout << Mcols << " " << Mrows2 << std::endl;
    std::cout << Mrowr << " " << Mcolr2 << std::endl;
    std::cout << Mcols << " " << Mcols2 << std::endl;
    std::exit(1);
  } 
  */
  ////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////

  return delta_entropy2;
} // end of compute_delta_entropy


template <typename W>
size_t Graph_P<W>::_carry_out_best_merges(
  const std::vector<size_t>& bestMerges,
  const std::vector<int>& best_merge_for_each_block,
  size_t B_to_merge,
  std::vector<size_t>& b,
  std::vector<size_t>& block_map,
  std::vector<size_t>& remaining_blocks) 
{
  size_t B = _num_blocks;
  
  block_map.clear();
  block_map.resize(B);
  std::iota(block_map.begin(), block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;

  while (num_merge < B_to_merge) {
    int mergeFrom = bestMerges[counter];
    int mergeTo = block_map[best_merge_for_each_block[bestMerges[counter]]];
    counter++;
    if (mergeTo != mergeFrom) {
      for (size_t i = 0; i < B; i++) {
        if (block_map[i] == mergeFrom) block_map[i] = mergeTo;
      }
      for (size_t i = 0; i < b.size(); i++) {
        if (b[i] == mergeFrom) b[i] = mergeTo;
      }
      num_merge += 1;
    }
  }
  
  remaining_blocks = _unique(b);
  block_map.clear();
  block_map.resize(B, -1);
  for (size_t i = 0; i < remaining_blocks.size(); i++) {
    block_map[remaining_blocks[i]] = i;
  }

  for (auto& it : b) {
    it = block_map[it];
  }

  return B - B_to_merge;
} // end of carry_out_best_merges

template <typename W>
float Graph_P<W>::_compute_overall_entropy(
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol, //TODO::?
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  const std::vector<W>& d_out,
  const std::vector<W>& d_in) 
{

  ////////////////////////////////////
  //////////////////////////////////
  //if (!_compare_if_M(Mrow, Mrow2)) {
  //  std::cout << "overall entropy block Mrow diff\n";
  //  std::exit(1);
  //}
  //////////////////////////////////
  //////////////////////////////

  /*
  float data_S = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow[i]) {
      if (w != 0) {
        data_S -= w * std::log(w / (float)(d_out[i] * d_in[v]));
      }
    }
  }
  */
  /////////////////////////////////////////////////////
  ////////////////////////////////////////////////////
  std::vector<W> tmp(_num_blocks);
  
  float data_S2 = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    std::fill(tmp.begin(), tmp.end(), 0);
    for (const auto& [v, w] : Mrow[i]) {
      tmp[v] += w;
    }
    for (size_t v = 0; v < _num_blocks; v++) {
      if (tmp[v] != 0) {
        data_S2 -= tmp[v] * std::log(tmp[v] / (float)(d_out[i] * d_in[v]));
      }
    }
  }
 
  //if (std::abs(data_S - data_S2) > 0.5) {
  //  std::cout << "overall_entropy error\n";
  //  std::cout << data_S << ", " << data_S2 << std::endl;
  //  std::exit(1);
  //}
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////


  float model_S_term = (float)_num_blocks*_num_blocks/_E;
  float model_S = (float)(_E * (1 + model_S_term) * log(1 + model_S_term)) - 
                          (model_S_term * log(model_S_term)) + (_N * log(_num_blocks));

  return model_S + data_S2;
} // end of compute_overall_entropy


template <typename W>
float Graph_P<W>::_compute_Hastings_correction(
  size_t s,
  size_t ni, 
  const std::vector<size_t>& partitions,
  //const std::vector< std::unordered_map<size_t, W> >& Mrow,
  //const std::vector< std::unordered_map<size_t, W> >& Mcol,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  const std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& d,
  const std::vector<W>& d_new) 
{
  float p_forward = 0;
  float p_backward = 0;
  //TODO: opt
  std::vector<W> Mcols(_num_blocks);
  std::vector<W> Mrows(_num_blocks);
  for (const auto& [v, w] : Mrow[s]) {
    Mrows[v] += w;
  }
  for (const auto& [v, w] : Mcol[s]) {
    Mcols[v] += w;
  }
  
  size_t b;
  for (const auto& [v, w] : _out_neighbors[ni]) {
    b = partitions[v];
    p_forward += (float)w*(Mcols[b] + Mrows[b] + 1)/(d[b]+_num_blocks);
    p_backward += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
  }
  for (const auto & [v, w] : _in_neighbors[ni]) {
    b = partitions[v];
    p_forward += (float)w*(Mcols[b] + Mrows[b] + 1)/(d[b]+_num_blocks);
    p_backward += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
  }



  ///////////////////////////////////////
  //////////////////////////////////////
  /*
  float p_forward2 = 0;
  float p_backward2 = 0;
  //TODO: opt
  std::vector<W> Mcols2(_num_blocks);
  std::vector<W> Mrows2(_num_blocks);
  for (const auto& [v, w] : Mrow2[s]) {
    Mrows2[v] += w;
  }
  for (const auto& [v, w] : Mcol2[s]) {
    Mcols2[v] += w;
  }

  for (const auto& [v, w] : _out_neighbors[ni]) {
    b = partitions[v];
    p_forward2 += (float)w*(Mcols2[b] + Mrows2[b] + 1)/(d[b]+_num_blocks);
    p_backward2 += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
  }
  for (const auto & [v, w] : _in_neighbors[ni]) {
    b = partitions[v];
    p_forward2 += (float)w*(Mcols2[b] + Mrows2[b] + 1)/(d[b]+_num_blocks);
    p_backward2 += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
  }
  
  if (std::abs(p_forward - p_forward2) > 1e-6) {
    std::cout << "Hastings_correction\n";
    std::cout << p_forward << ", " << p_forward2 << std::endl;
    std::exit(1);
  }
  if (std::abs(p_backward - p_backward2) > 1e-6) {
    std::cout << "Hastings_correction\n";
    std::cout << p_backward << ", " << p_backward2 << std::endl;
    std::exit(1);
  }
  */
  /////////////////////////////////////
  /////////////////////////////////////

  return p_backward / p_forward;
} // end of compute_Hastings_correction


template <typename W>
void Graph_P<W>::_update_partition(
  size_t ni,
  size_t r,
  size_t s, 
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_s_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& M_s_col,
  const std::vector<W>& d_out_new,
  const std::vector<W>& d_in_new,
  const std::vector<W>& d_new,
  std::vector<size_t>& b,
  //std::vector< std::unordered_map<size_t, W> >& Mrow,
  //std::vector< std::unordered_map<size_t, W> >& Mcol,
  std::vector< std::vector<std::pair<size_t, W>> >& Mrow,
  std::vector< std::vector<std::pair<size_t, W>> >& Mcol,
  std::vector<W>& d_out,
  std::vector<W>& d_in,
  std::vector<W>& d) 
{
 ///////////////////////////////////////////
 /*
  std::vector< std::unordered_map<size_t, W> > f(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow2[i]) {
      f[i][v] += w;
    }
  }
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow[i]) {
      if (w != f[i][v]) {
        std::cout << "Mrow check error update before\n";
        std::exit(1);
      }
    }
  }

  std::vector< std::unordered_map<size_t, W> > f2(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mcol2[i]) {
      f2[i][v] += w;
    }
  }
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mcol[i]) {
      if (w != f2[i][v]) {
        std::cout << "Mcol check error update before\n";
        std::exit(1);
      }
    }
  }
*/
 
 /////////////////////////////////////////////
  ////////////////////////////////////////////
  //check before update
  b[ni] = s;
  // TODO: why can I just clear the row and col and only update the non zero
  // term?
  //

  /*
  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_row[i] != 0) {
      Mrow[r][i] = M_r_row[i];
      Mcol[i][r] = M_r_row[i];
    }
    else {
      Mrow[r].erase(i);
      Mcol[i].erase(r);
    }
    if (M_s_row[i] != 0) {
      Mrow[s][i] = M_s_row[i];
      Mcol[i][s] = M_s_row[i];
    }
    else {
      Mrow[s].erase(i);
      Mcol[i].erase(s);
    }
    if (M_r_col[i] != 0) {
      Mrow[i][r] = M_r_col[i];
      Mcol[r][i] = M_r_col[i];
    }
    else {
      Mrow[i].erase(r);
      Mcol[r].erase(i);
    }
    if (M_s_col[i] != 0) {
      Mrow[i][s] = M_s_col[i];
      Mcol[s][i] = M_s_col[i];
    }
    else {
      Mrow[i].erase(s);
      Mcol[s].erase(i);
    }
  }
  */
  // from the differet perspective
  Mrow[r].clear();
  Mrow[s].clear();
  Mcol[r].clear();
  Mcol[s].clear();
  for (size_t i = 0; i < _num_blocks; i++) {
    /*
    Mrow2[i].erase(
        std::remove_if(Mrow2[i].begin(), Mrow2[i].end(), [r, s](const std::pair<size_t, W>& p) {
            return (p.first == r) || (p.first == s);
        }),
        Mrow2[i].end()
    );
    Mcol2[i].erase(
        std::remove_if(Mcol2[i].begin(), Mcol2[i].end(), [r, s](const std::pair<size_t, W>& p) {
            return (p.first == r) || (p.first == s);
        }),
        Mcol2[i].end()
    );
    if (M_r_row[i] != 0) {
      Mrow2[r].emplace_back(std::make_pair(i, M_r_row[i]));
      Mcol2[i].emplace_back(std::make_pair(r, M_r_row[i]));
    }
    if (M_s_row[i] != 0) {
      Mrow2[s].emplace_back(std::make_pair(i, M_s_row[i]));
      Mcol2[i].emplace_back(std::make_pair(s, M_s_row[i]));
    }
    if (i != r && i != s) { 
      if (M_r_col[i] != 0) {
        Mrow2[i].emplace_back(std::make_pair(r, M_r_col[i]));
        Mcol2[r].emplace_back(std::make_pair(i, M_r_col[i]));
      }
      if (M_s_col[i] != 0) {
        Mrow2[i].emplace_back(std::make_pair(s, M_s_col[i]));
        Mcol2[s].emplace_back(std::make_pair(i, M_s_col[i]));
      }  
    }
    */
    if (M_r_row[i] != 0) {
      Mrow[r].emplace_back(std::make_pair(i, M_r_row[i]));
    }
    if (M_s_row[i] != 0) {
      Mrow[s].emplace_back(std::make_pair(i, M_s_row[i]));
    }
    if (i != r && i != s) { 
       Mrow[i].erase(
        std::remove_if(Mrow[i].begin(), Mrow[i].end(), [r, s](const std::pair<size_t, W>& p) { 
            return (p.first == r) || (p.first == s);
        }),  
        Mrow[i].end()
      ); 
  
      if (M_r_col[i] != 0) {
        Mrow[i].emplace_back(std::make_pair(r, M_r_col[i]));
      }
      if (M_s_col[i] != 0) {
        Mrow[i].emplace_back(std::make_pair(s, M_s_col[i]));
      }  
    }
  } 

  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_col[i] != 0) {
      Mcol[r].emplace_back(std::make_pair(i, M_r_col[i]));
    }
    if (M_s_col[i] != 0) {
      Mcol[s].emplace_back(std::make_pair(i, M_s_col[i]));
    }
    if (i != r && i != s) {
      Mcol[i].erase(
        std::remove_if(Mcol[i].begin(), Mcol[i].end(), [r, s](const std::pair<size_t, W>& p) {
            return (p.first == r) || (p.first == s);
        }),
        Mcol[i].end()
      );
      if (M_r_row[i] != 0) {
        Mcol[i].emplace_back(std::make_pair(r, M_r_row[i]));
      }
      if (M_s_row[i] != 0) {
        Mcol[i].emplace_back(std::make_pair(s, M_s_row[i]));
      }     
    }
  
  }

  ////////////////////////////////////////
  ///////////////////////////////////////
  /*
  f.clear();
  f.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow2[i]) {
      f[i][v] += w;
    }
  }
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mrow[i]) {
      if (w != f[i][v]) {
        std::cout << "Mrow check error update\n";
        std::cout << "Mrow= " << w << " Mrow2=" << f[i][v] << "\n";
        std::cout << "v=" << v << " r=" << r << " s=" << s << "\n";
        std::exit(1);
      }
    }
  }

  f2.clear();
  f2.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mcol2[i]) {
      f2[i][v] += w;
    }
  }
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : Mcol[i]) {
      if (w != f2[i][v]) {
        std::cout << "Mcol check error update\n";
        std::exit(1);
      }
    }
  }
  */
  // remove all the zero in the vector

  // I need to know where is converted to zero!!!
  // solution:
  // compare the M and the r/s row/col (which can be done in the previous
  // I need to know where is converted to zero!!!
  // solution:
  // compare the M and the r/s row/col (which can be done in the previous
  // function)
  // record the change below
 


  d_out = d_out_new;
  d_in = d_in_new;
  d = d_new;

} // end of update_partition

template <typename W>
bool Graph_P<W>::_prepare_for_partition_next(
  float S,
  std::vector<size_t>& b,
  //std::vector< std::unordered_map<size_t, W> >& Mrow,
  //std::vector< std::unordered_map<size_t, W> >& Mcol,
  std::vector< std::vector<std::pair<size_t, W>> >& Mrow2,
  std::vector< std::vector<std::pair<size_t, W>> >& Mcol2,
  std::vector<W>& d,
  std::vector<W>& d_out,
  std::vector<W>& d_in,
  size_t& B,
  size_t& B_to_merge,
  Old& old,
  float B_rate
) {
  //TODO: 
  // do i need to pass B and old??
  
  bool optimal_B_found = false;
  int index;
  
  if (S <= old.overall_entropy_med) {  // if the current overall entropy is the best so far
    if ( old.num_blocks_med > B) { 
      old.partitions_large = old.partitions_med;
      //old.Mrow_large = old.Mrow_med;
      //old.Mcol_large = old.Mcol_med;
      //
      old.Mrow_large2 = old.Mrow_med2;
      old.Mcol_large2 = old.Mcol_med2;   
      //
      old.block_degree_large = old.block_degree_med;
      old.block_degree_in_large = old.block_degree_in_med;
      old.block_degree_out_large = old.block_degree_out_med;
      old.overall_entropy_large = old.overall_entropy_med;
      old.num_blocks_large = old.num_blocks_med;
    }
    else {
      old.partitions_small = old.partitions_med;
      //old.Mrow_small = old.Mrow_med;
      //old.Mcol_small = old.Mcol_med;
      //
      old.Mrow_small2 = old.Mrow_med2;
      old.Mcol_small2 = old.Mcol_med2;
      //
      old.block_degree_small = old.block_degree_med;
      old.block_degree_in_small = old.block_degree_in_med;
      old.block_degree_out_small = old.block_degree_out_med;
      old.overall_entropy_small = old.overall_entropy_med;
      old.num_blocks_small = old.num_blocks_med;  
    }
    old.partitions_med = b; 
    //old.Mrow_med = Mrow;
    //old.Mcol_med = Mcol;
    //
    old.Mrow_med2 = Mrow2;
    old.Mcol_med2 = Mcol2;
    //
    old.block_degree_med = d;
    old.block_degree_in_med = d_in;
    old.block_degree_out_med = d_out;
    old.overall_entropy_med = S;
    old.num_blocks_med = B;
  }
  else {
    if ( old.num_blocks_med > B) {
      old.partitions_small = b;
      //old.Mrow_small = Mrow;
      //old.Mcol_small = Mcol;
      //
      old.Mrow_small2 = Mrow2;
      old.Mcol_small2 = Mcol2;
      //
      old.block_degree_small = d;
      old.block_degree_in_small = d_in;
      old.block_degree_out_small = d_out;
      old.overall_entropy_small = S;
      old.num_blocks_small = B; 
    }
    else {
      old.partitions_large = b;
      //old.Mrow_large = Mrow;
      //old.Mcol_large = Mcol;
      //
      old.Mrow_large2 = Mrow2;
      old.Mcol_large2 = Mcol2;
      //
      old.block_degree_large = d;
      old.block_degree_in_large = d_in;
      old.block_degree_out_large = d_out;
      old.overall_entropy_large = S;
      old.num_blocks_large = B;
    }
  }
 
  if (std::isinf(old.overall_entropy_small)) {
    B_to_merge = (int)B*B_rate;
    if (B_to_merge == 0) optimal_B_found = true;
    b = old.partitions_med;
    //Mrow = old.Mrow_med;
    //Mcol = old.Mcol_med;
    //
    Mrow2 = old.Mrow_med2;
    Mcol2 = old.Mcol_med2;
    ///
    d = old.block_degree_med;
    d_out = old.block_degree_out_med;
    d_in = old.block_degree_in_med; 
  }
  else {
    // golden ratio search bracket established
    // we have found the partition with the optimal number of blocks
    if (old.num_blocks_large - old.num_blocks_small == 2) {
      optimal_B_found = true;
      B = old.num_blocks_med;
      b = old.partitions_med;
    }
    // not done yet, find the next number of block to try according to the golden ratio search
    // the higher segment in the bracket is bigger
    else {
      // the higher segment in the bracket is bigger
      if ((old.num_blocks_large-old.num_blocks_med) >= (old.num_blocks_med-old.num_blocks_small)) {  
        int next_B_to_try = old.num_blocks_med + 
          static_cast<int>(round((old.num_blocks_large - old.num_blocks_med) * 0.618));
        B_to_merge = old.num_blocks_large - next_B_to_try;
        B = old.num_blocks_large;
        b = old.partitions_large;
        //Mrow = old.Mrow_large;
        //Mcol = old.Mcol_large;
        //
        Mrow2 = old.Mrow_large2;
        Mcol2 = old.Mcol_large2;
        //
        d = old.block_degree_large;
        d_out = old.block_degree_out_large;
        d_in = old.block_degree_in_large;
      }
      else {
        int next_B_to_try = old.num_blocks_small 
          + static_cast<int>(round((old.num_blocks_med - old.num_blocks_small) * 0.618));
        B_to_merge = old.num_blocks_med - next_B_to_try;
        B = old.num_blocks_med;
        b = old.partitions_med;
        //Mrow = old.Mrow_med;
        //Mcol = old.Mcol_med;
        //
        Mrow2 = old.Mrow_med2;
        Mcol2 = old.Mcol_med2;
        //
        d = old.block_degree_med;
        d_out = old.block_degree_out_med;
        d_in = old.block_degree_in_med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


