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

    size_t _N;                                      // number of node
    size_t _E;                                      // number of edge
    
    std::vector<Edge> _edges;
    std::vector<std::vector<std::pair<size_t, W>>> _out_neighbors;
    std::vector<std::vector<std::pair<size_t, W>>> _in_neighbors;

    size_t _num_blocks;

    std::random_device _rd;
    std::default_random_engine _generator;
   
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

    // for golden ratio bracket
    struct Old {
      std::vector<size_t> partitions_large;
      std::vector<size_t> partitions_med;
      std::vector<size_t> partitions_small;
      std::vector<W> interblock_edge_count_large;
      std::vector<W> interblock_edge_count_med;
      std::vector<W> interblock_edge_count_small;
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

    // functions used internally
    void _initialize_edge_counts(
      const std::vector<size_t>& partitions,
      std::vector<W>& M, 
      std::vector<W>& d_out, 
      std::vector<W>& d_in, 
      std::vector<W>& d
    );
 
    void _propose_new_partition_block(
      size_t r,
      const std::vector<size_t>& partitions,
      const std::vector<W>& M,
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
      const std::vector<W>& M,
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
      const std::vector<W>& M,
      size_t r,
      size_t s,
      std::vector<W>& M_r_row,
      std::vector<W>& M_s_row,
      std::vector<W>& M_r_col,
      std::vector<W>& M_s_col
    );

    void _compute_new_rows_cols_interblock_edge_count_nodal(
      const std::vector<W>& M,
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
      const std::vector<W>& M,
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
      const std::vector<W>& M,
      const std::vector<W>& d_out,
      const std::vector<W>& d_in
    ); 

    float _compute_Hastings_correction(
      size_t s,
      size_t ni,
      const std::vector<size_t>& partitions,
      const std::vector<W>& M,
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
      std::vector<W>& M,
      std::vector<W>& d_out,
      std::vector<W>& d_in,
      std::vector<W>& d
    );

    bool _prepare_for_partition_next(
      float S,
      std::vector<size_t>& b,
      std::vector<W>& M,
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
  
  std::vector<W> interblock_edge_count;
  std::vector<W> block_degrees_out;
  std::vector<W> block_degrees_in;
  std::vector<W> block_degrees;

  _initialize_edge_counts(
    partitions, 
    interblock_edge_count, 
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
        &interblock_edge_count,
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
              interblock_edge_count,
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
              interblock_edge_count,
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
              interblock_edge_count,
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

    interblock_edge_count.clear();
    block_degrees_out.clear();
    block_degrees_in.clear();
    block_degrees.clear();
    _initialize_edge_counts(
      partitions, 
      interblock_edge_count, 
      block_degrees_out, 
      block_degrees_in, 
      block_degrees
    );

    int total_num_nodal_moves = 0;
    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);

    float overall_entropy = _compute_overall_entropy(
                              interblock_edge_count, 
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
          interblock_edge_count,
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
            interblock_edge_count,
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
                                    interblock_edge_count,
                                    interblock_edge_count_r_row_new,
                                    interblock_edge_count_r_col_new,
                                    block_degrees,
                                    block_degrees_new
                                  );
          } // calculate Hastings_correction
          
          float delta_entropy = _compute_delta_entropy(
                                  current_block,
                                  proposal,
                                  interblock_edge_count,
                                  interblock_edge_count_r_row_new,
                                  interblock_edge_count_s_row_new,
                                  interblock_edge_count_r_col_new,
                                  interblock_edge_count_s_col_new,
                                  block_degrees_out,
                                  block_degrees_in,
                                  block_degrees_out_new,
                                  block_degrees_in_new
                                );

          float p_accept = std::min(static_cast<float>(std::exp(-beta * delta_entropy))
                                                       * Hastings_correction, 1.0f);


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
              interblock_edge_count,
              block_degrees_out,
              block_degrees_in,
              block_degrees
            );
          }
        } // end if 
      } // end current_node

      float oe = overall_entropy = _compute_overall_entropy(
                                     interblock_edge_count, 
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
                        interblock_edge_count, 
                        block_degrees_out, 
                        block_degrees_in
                      );

    if (verbose)
      printf("Total number of nodal moves: %d, overall_entropy: %.5f\n", 
              total_num_nodal_moves, overall_entropy);


    optimal_num_blocks_found = _prepare_for_partition_next(
                                 overall_entropy,
                                 partitions,
                                 interblock_edge_count,
                                 block_degrees,
                                 block_degrees_out,
                                 block_degrees_in,
                                 _num_blocks,
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
  std::vector<W>& M, 
  std::vector<W>& d_out, 
  std::vector<W>& d_in, 
  std::vector<W>& d
) {

  _taskflow.clear();
  M.clear();
  M.resize(_num_blocks * _num_blocks, 0);

  // compute the initial interblock edge count
  for (size_t node = 0; node < _out_neighbors.size(); node++) {
    if (_out_neighbors[node].size() > 0) {
      size_t k1 = partitions[node];
      for (auto& e: _out_neighbors[node]) {
        M[k1*_num_blocks + partitions[e.first]] += e.second;
      }
    }
  }

  // compute initial block degrees
  d_out.clear();
  d_out.resize(_num_blocks, 0);
  d_in.clear();
  d_in.resize(_num_blocks, 0);
  d.clear();
  d.resize(_num_blocks, 0);
  for (size_t i = 0; i < _num_blocks; i++) {
    _taskflow.emplace([i, this, &M, &d_in, &d_out, &d] () {
      for (size_t j = 0; j < _num_blocks; j++) {
        d_out[i] += M[i*_num_blocks + j]; // row
        d_in[i] += M[j*_num_blocks + i]; // col
      }
      d[i] = d_out[i] + d_in[i];
    });
  }
  _executor.run(_taskflow).wait();
} // end of initialize_edge_counts

template <typename W>
void Graph_P<W>::_propose_new_partition_block(
  size_t r,
  const std::vector<size_t>& partitions,
  const std::vector<W>& M,
  const std::vector<W>& d,
  const std::default_random_engine& generator,
  size_t& s,
  W& k_out,
  W& k_in,
  W& k,
  std::vector<size_t>& neighbors,
  std::vector<float>& prob)
{
  k_out = 0;
  k_in = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    if (M[_num_blocks*r + i] != 0) {
      k_out += M[_num_blocks*r + i];
      neighbors.emplace_back(i);
      prob.emplace_back((float)M[_num_blocks*r + i]);
    }
    if (M[_num_blocks*i + r] != 0) {
      k_in += M[_num_blocks*i + r];
      neighbors.emplace_back(i);
      prob.emplace_back((float)M[_num_blocks*i + r]);
    }
  }
  k = k_out + k_in;
  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if ( k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  for (auto& p : prob) p/=k;
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(generator))];
  size_t u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(generator));
  if ( x <= (float)_num_blocks/(d[u]+_num_blocks) ) {
    neighbors.clear();  
    neighbors.resize(_num_blocks); 
    std::iota(neighbors.begin(), neighbors.end(), 0);
    neighbors.erase(neighbors.begin() + r);
    std::uniform_int_distribution<int> choice(0, neighbors.size()-1);
    s = neighbors[choice(const_cast<std::default_random_engine&>(generator))];
  }
  else {
    prob.clear();
    float multinomial_prob_sum = 0;
    for (size_t i = 0; i < _num_blocks; i++) {
      prob.emplace_back((float)(M[u*_num_blocks + i] + M[i*_num_blocks + u])/d[u]);
      multinomial_prob_sum += prob[i];
    }
    prob[r] = 0;
    if (multinomial_prob_sum == 0) {
      neighbors.clear();
      std::iota(neighbors.begin(), neighbors.end(), 0);
      neighbors.erase(neighbors.begin() + r);
      std::uniform_int_distribution<int> choice(0, neighbors.size()-1);
      s = neighbors[choice(const_cast<std::default_random_engine&>(generator))];
      return;
    }
    else {
      for (auto& it : prob) it /= multinomial_prob_sum;
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
  const std::vector<W>& M,
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

  for (const auto & out : _out_neighbors[ni]) {
    neighbors.emplace_back(out.first);
    prob.emplace_back((float)out.second);
    k_out += out.second;
  }
  for (const auto & in : _in_neighbors[ni]) {
    neighbors.emplace_back(in.first);
    prob.emplace_back((float)in.second);
    k_in += in.second;
  }
  k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if (k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  for (auto& p : prob) p/=k;
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
    float multinomial_prob_sum = 0;
    for (size_t i = 0; i < _num_blocks; i++) {
      prob.emplace_back((float)(M[u*_num_blocks + i] + M[i*_num_blocks + u])/d[u]);
      multinomial_prob_sum += prob[i];
    }
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }
}

template <typename W>
void Graph_P<W>::_compute_new_rows_cols_interblock_edge_count_block(
  const std::vector<W>& M,
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
  for (size_t i = 0; i < _num_blocks; i++) {
    M_r_row[i] = 0;
    M_r_col[i] = 0;
    M_s_row[i] = M[s*_num_blocks + i];
    M_s_col[i] = M[i*_num_blocks + s]; 
    // out_blocks
    if (M[_num_blocks*r + i] != 0) {
      if (i == s) {
        count_out_sum_s += M[_num_blocks*r + i];
      }
      M_s_row[i] += M[_num_blocks*r + i];
    }
    // in_blocks
    if (M[_num_blocks*i + r] != 0) {
      if (i == s) {
        count_in_sum_s += M[_num_blocks*i + r];
      }
      M_s_col[i] += M[_num_blocks*i + r];
    }
  }

  M_s_row[r] -= count_in_sum_s;
  M_s_row[s] += count_in_sum_s;
  M_s_row[r] -= M[r*_num_blocks + r];
  M_s_row[s] += M[r*_num_blocks + r];
  
  M_s_col[r] -= count_out_sum_s;
  M_s_col[s] += count_out_sum_s;
  M_s_col[r] -= M[r*_num_blocks + r];
  M_s_col[s] += M[r*_num_blocks + r];
}


template <typename W>
void Graph_P<W>::_compute_new_rows_cols_interblock_edge_count_nodal(
  const std::vector<W>& M,
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

  for (size_t i = 0; i < _num_blocks; i++) {
    M_r_row[i] = M[r*_num_blocks + i];
    M_r_col[i] = M[i*_num_blocks + r];
    M_s_row[i] = M[s*_num_blocks + i];
    M_s_col[i] = M[i*_num_blocks + s];
  } 
  // out_blocks
  for (const auto& out : _out_neighbors[ni]) {
    if (partitions[out.first] == r) {
      count_out_sum_r += out.second;
    }
    if (partitions[out.first] == s) {
      count_out_sum_s += out.second;
    }
    if (out.first == ni) {
      count_self += out.second;
    }
    M_r_row[ partitions[out.first] ] -= out.second;
    M_s_row[ partitions[out.first] ] += out.second;
  }
  // in_blocks
  for (const auto& in : _in_neighbors[ni]) {
    if (partitions[in.first] == r) {
      count_in_sum_r += in.second;
    }
    if (partitions[in.first] == s) {
      count_in_sum_s += in.second;
    }
    M_r_col[ partitions[in.first] ] -= in.second;
    M_s_col[ partitions[in.first] ] += in.second;
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
  std::vector<W>& d_new
) {

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
  const std::vector<W>& M,
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_s_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& M_s_col,
  const std::vector<W>& d_out,
  const std::vector<W>& d_in,
  const std::vector<W>& d_out_new,
  const std::vector<W>& d_in_new
) {
  
  size_t B = _num_blocks;

  float delta_entropy = 0;
  for (size_t i = 0; i < B; i++) {
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
    // M_r_t1[i] = M[r*B + i];
    if (M[r*B + i] != 0) {
      delta_entropy += M[r*B + i] * std::log(static_cast<float>
                                      (M[r*B + i]) / (d_in[i] * d_out[r])
                                    );
    }
    // M_s_t1[i] = M[s*B + i]
    if (M[s*B + i] != 0) {
      delta_entropy += M[s*B + i] * std::log(static_cast<float>
                                      (M[s*B + i]) / (d_in[i] * d_out[s])
                                    );
    }
    // avoid duplicate counting
    if (i != r && i != s) {
      // M_t2_r[i] = M[i*B + r]
      if (M[i*B + r] != 0) {
        delta_entropy += M[i*B + r] * std::log(static_cast<float>
                                        (M[i*B + r]) / (d_out[i] * d_in[r])
                                      );
      }
      // M_t2_s[i] = M[i*B + s]
      if (M[i*B + s] != 0) {
        delta_entropy += M[i*B + s] * std::log(static_cast<float>
                                        (M[i*B + s]) / (d_out[i] * d_in[s])
                                      );
      }
    }
  }

  return delta_entropy;
} // end of compute_delta_entropy


template <typename W>
size_t Graph_P<W>::_carry_out_best_merges(
  const std::vector<size_t>& bestMerges,
  const std::vector<int>& best_merge_for_each_block,
  size_t B_to_merge,
  std::vector<size_t>& b,
  std::vector<size_t>& block_map,
  std::vector<size_t>& remaining_blocks
) {
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
  const std::vector<W>& M,
  const std::vector<W>& d_out,
  const std::vector<W>& d_in
) {
  size_t B = _num_blocks;
  
  float data_S = 0;
  for (size_t i = 0; i < B; i++) { 
    for (size_t j = 0; j < B; j++) {
      if (M[i*B + j] != 0) {
        data_S -= M[i*B + j] * log(M[i*B + j] / (float)(d_out[i] * d_in[j]));
      }
    }
  }

  float model_S_term = (float)B*B/_E;
  float model_S = (float)(_E * (1 + model_S_term) * log(1 + model_S_term)) - 
                          (model_S_term * log(model_S_term)) + (_N * log(B));

  return model_S + data_S;

} // end of compute_overall_entropy


template <typename W>
float Graph_P<W>::_compute_Hastings_correction(
  size_t s,
  size_t ni, 
  const std::vector<size_t>& partitions,
  const std::vector<W>& M,
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& d,
  const std::vector<W>& d_new) 
{
  float p_forward = 0;
  float p_backward = 0;
  for (const auto& out : _out_neighbors[ni]) {
    size_t block = partitions[out.first];
    p_forward += (float)out.second * (M[block*_num_blocks + s] + M[s*_num_blocks 
      + block] + 1) / (d[block] + _num_blocks);
    p_backward += (float)out.second * (M_r_row[block] + M_r_col[block] + 1)
      / (d_new[block] + _num_blocks);
  }
  for (const auto& in : _in_neighbors[ni]) {
    size_t block = partitions[in.first];
    p_forward += (float)in.second * (M[block*_num_blocks + s] + M[s*_num_blocks +
      block] + 1) / (d[block] + _num_blocks);
    p_backward += (float)in.second * (M_r_row[block] + M_r_col[block] + 1)
      / (d_new[block] + _num_blocks);
  } 
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
  std::vector<W>& M,
  std::vector<W>& d_out,
  std::vector<W>& d_in,
  std::vector<W>& d
) {
  size_t B = _num_blocks;
  b[ni] = s;
  for (size_t i = 0; i < B; i++) {
    M[r*B + i] = M_r_row[i];
    M[s*B + i] = M_s_row[i];
    M[i*B + r] = M_r_col[i];
    M[i*B + s] = M_s_col[i];
  }
  d_out = d_out_new;
  d_in = d_in_new;
  d = d_new;

} // end of update_partition

template <typename W>
bool Graph_P<W>::_prepare_for_partition_next(
  float S,
  std::vector<size_t>& b,
  std::vector<W>& M,
  std::vector<W>& d,
  std::vector<W>& d_out,
  std::vector<W>& d_in,
  size_t& B,
  size_t& B_to_merge,
  Old& old,
  float B_rate
) {
  bool optimal_B_found = false;
  int index;
  
  if (S <= old.overall_entropy_med) {  // if the current overall entropy is the best so far
    if ( old.num_blocks_med > B) { 
      old.partitions_large = old.partitions_med;
      old.interblock_edge_count_large = old.interblock_edge_count_med;
      old.block_degree_large = old.block_degree_med;
      old.block_degree_in_large = old.block_degree_in_med;
      old.block_degree_out_large = old.block_degree_out_med;
      old.overall_entropy_large = old.overall_entropy_med;
      old.num_blocks_large = old.num_blocks_med;
    }
    else {
      old.partitions_small = old.partitions_med;
      old.interblock_edge_count_small = old.interblock_edge_count_med;
      old.block_degree_small = old.block_degree_med;
      old.block_degree_in_small = old.block_degree_in_med;
      old.block_degree_out_small = old.block_degree_out_med;
      old.overall_entropy_small = old.overall_entropy_med;
      old.num_blocks_small = old.num_blocks_med;  
    }
    old.partitions_med = b; 
    old.interblock_edge_count_med = M;
    old.block_degree_med = d;
    old.block_degree_in_med = d_in;
    old.block_degree_out_med = d_out;
    old.overall_entropy_med = S;
    old.num_blocks_med = B;
  }
  else {
    if ( old.num_blocks_med > B) {
      old.partitions_small = b;
      old.interblock_edge_count_small = M;
      old.block_degree_small = d;
      old.block_degree_in_small = d_in;
      old.block_degree_out_small = d_out;
      old.overall_entropy_small = S;
      old.num_blocks_small = B; 
    }
    else {
      old.partitions_large = b;
      old.interblock_edge_count_large = M;
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
    M = old.interblock_edge_count_med;
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
        M = old.interblock_edge_count_large;
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
        M = old.interblock_edge_count_med;
        d = old.block_degree_med;
        d_out = old.block_degree_out_med;
        d_in = old.block_degree_in_med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


