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
#include <mutex>
#include <condition_variable>
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
    void partition();    
    void partition_tf();
    const size_t& num_nodes() const { return _N; }
    const size_t& num_edges() const { return _E; }
    const std::vector<size_t>& get_partitions() const { return _partitions; }

    // constructor
    Graph_P(const std::string& FileName, 
      //size_t num_threads = std::thread::hardware_concurrency()) :
      size_t num_threads = 1) :
      _executor(num_threads),
      _pt_probabilities(num_threads),
      _pt_neighbors(num_threads),
      _pt_interblock_edge_count_r_row_new(num_threads),
      _pt_interblock_edge_count_s_row_new(num_threads),
      _pt_interblock_edge_count_r_col_new(num_threads),
      _pt_interblock_edge_count_s_col_new(num_threads),
      _pt_block_degrees_out_new(num_threads),
      _pt_block_degrees_in_new(num_threads),
      _pt_block_degrees_new(num_threads),
      _pt_r_row(num_threads),
      _pt_s_row(num_threads),
      _pt_r_col(num_threads),
      _pt_s_col(num_threads),
      _pt_num_nodal_move_itr(num_threads),
      _pt_delta_entropy_itr(num_threads)
    {
      load_graph_from_tsv(FileName);
      _generator.seed(_rd());
    }

    // partition ground truth
    std::vector<size_t> truePartitions;

  private:

    size_t _N; // number of node
    size_t _E; // number of edge
    size_t _num_blocks; 
    size_t _num_blocks_to_merge;

    std::vector<Edge> _edges;
    std::vector<std::vector<std::pair<size_t, W>>> _out_neighbors;
    std::vector<std::vector<std::pair<size_t, W>>> _in_neighbors;

    std::vector<size_t> _partitions;

    // taskflow   
    tf::Executor _executor;
    tf::Taskflow _taskflow;
          
    std::vector< std::vector<float>>  _pt_probabilities;
    std::vector< std::vector<size_t>> _pt_neighbors;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_r_row_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_s_row_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_r_col_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_s_col_new;
    std::vector< std::vector<W>>      _pt_block_degrees_out_new;
    std::vector< std::vector<W>>      _pt_block_degrees_in_new;
    std::vector< std::vector<W>>      _pt_block_degrees_new;
    std::vector< std::vector<W>>      _pt_r_row;
    std::vector< std::vector<W>>      _pt_s_row;
    std::vector< std::vector<W>>      _pt_r_col;
    std::vector< std::vector<W>>      _pt_s_col;
    std::vector< int >                _pt_num_nodal_move_itr;
    std::vector< float >              _pt_delta_entropy_itr;

    // save data for golden ratio bracket
    struct Old {
      std::vector<size_t> partitions_large;
      std::vector<size_t> partitions_med;
      std::vector<size_t> partitions_small;
      std::vector<W> M_large;
      std::vector<W> M_med;
      std::vector<W> M_small;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_large;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_med;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow_small;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_large;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_med;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol_small;
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

      Old() : overall_entropy_large(std::numeric_limits<float>::infinity()), 
              overall_entropy_med(std::numeric_limits<float>::infinity()),
              overall_entropy_small(std::numeric_limits<float>::infinity()),
              num_blocks_large(0),
              num_blocks_med(0),
              num_blocks_small(0) {}
    };
  
    Old _old;  
    std::random_device _rd;
    std::default_random_engine _generator;
    std::vector<W> _M;
    std::vector< std::vector<std::pair<size_t, W>> > _Mrow;
    std::vector< std::vector<std::pair<size_t, W>> > _Mcol;
    std::vector<W> _block_degrees_out;
    std::vector<W> _block_degrees_in;
    std::vector<W> _block_degrees;
    std::vector<size_t> _bestMerges;
    std::vector<size_t> _remaining_blocks;
    std::set<size_t> _seen;

    // functions used internally
    void _initialize_edge_counts();
 
    void _propose_new_partition_block(
      size_t r,
      const std::vector<size_t>& partitions,
      size_t& s,
      W& k_out,
      W& k_in,
      W& k,
      std::vector<float>& prob
    );
 
    void _propose_new_partition_nodal(
      size_t r,
      size_t ni,
      size_t& s,
      W& k_out,
      W& k_in,
      W& k,
      std::vector<size_t>& neighbors,
      std::vector<float>& prob
    );

    void _compute_new_rows_cols_interblock_edge_count_block(
      size_t r,
      size_t s,
      std::vector<W>& M_r_row,
      std::vector<W>& M_s_row,
      std::vector<W>& M_r_col,
      std::vector<W>& M_s_col
    );

    void _compute_new_rows_cols_interblock_edge_count_nodal(
      size_t r,
      size_t s,
      size_t ni,
      std::vector<W>& M_r_row,
      std::vector<W>& M_s_row,
      std::vector<W>& M_r_col,
      std::vector<W>& M_s_col
    );
    
    void _compute_new_block_degree(
      size_t r,
      size_t s,
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
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_s_row,
      const std::vector<W>& M_r_col,
      const std::vector<W>& M_s_col,
      const std::vector<W>& d_out_new,
      const std::vector<W>& d_in_new,
      std::vector<W>& r_row,
      std::vector<W>& s_row,
      std::vector<W>& r_col,
      std::vector<W>& s_col
    );        
     
    void _carry_out_best_merges(
      const std::vector<size_t>& best_merge_for_each_block,
      std::vector<size_t>& block_map
    );

    float _compute_overall_entropy(
      std::vector<W>& M_r_row  
    ); 

    float _compute_Hastings_correction(
      size_t s,
      size_t ni,
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_r_col,
      const std::vector<W>& d_new,
      std::vector<W>& Mrows,
      std::vector<W>& Mcols
    );

    bool _prepare_for_partition_next(
      float S,
      float B_rate
    );

    // utility functions
    void _argsort(const std::vector<float>& arr) {
      _bestMerges.clear();
      _bestMerges.resize(arr.size());
      std::iota(_bestMerges.begin(), _bestMerges.end(), 0);
      std::sort(_bestMerges.begin(), _bestMerges.end(),
                [&arr](int i, int j){ return arr[i] < arr[j]; });
    }

    void _unique(const std::vector<size_t>& arr) {
      _seen.clear();
      for (const auto& elem : arr) {
        if (_seen.find(elem) == _seen.end()) {
          _seen.insert(elem);
        }   
      }
      _remaining_blocks.clear();
      _remaining_blocks.insert(_remaining_blocks.end(), _seen.begin(), _seen.end());
      std::sort(_remaining_blocks.begin(), _remaining_blocks.end());
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
void Graph_P<W>::partition() {

  _num_blocks = _N;
  _partitions.clear();
  _partitions.resize(_num_blocks);
  std::iota(_partitions.begin(), _partitions.end(), 0);
  _num_blocks_to_merge = (size_t)_num_blocks * num_block_reduction_rate;

  std::vector<size_t> best_merge_for_each_block;
  std::vector<float> delta_entropy_for_each_block;
  std::vector<size_t> block_map;
  std::vector<size_t> block_partition;
  std::vector<W> interblock_edge_count_r_row_new;
  std::vector<float> itr_delta_entropy;
  int total_num_nodal_moves;
  int itr;
  bool optimal_num_blocks_found;

  tf::Task init = _taskflow.emplace([this] () {
    _initialize_edge_counts();
  }).name("init");

  tf::Task block_merge_init = _taskflow.emplace([this,
    &best_merge_for_each_block,
    &delta_entropy_for_each_block,
    &block_partition] (tf::Subflow& subflow) {
      best_merge_for_each_block.clear();
      best_merge_for_each_block.resize(_num_blocks, -1);
      delta_entropy_for_each_block.clear();
      delta_entropy_for_each_block.resize(_num_blocks, 
        std::numeric_limits<float>::infinity());
      block_partition.clear();
      block_partition.resize(_num_blocks, 0);
      std::iota(block_partition.begin(), block_partition.end(), 0);
  }).name("block_merge_init");

  tf::Task block_merge_par = _taskflow.for_each_index(0, int(_num_blocks), 1,
    [this, &block_partition,
    &best_merge_for_each_block,
    &delta_entropy_for_each_block] (int current_block) {    
      auto wid = _executor.this_worker_id();
      auto& prob = _pt_probabilities[wid];
      auto& interblock_edge_count_r_row_new = _pt_interblock_edge_count_r_row_new[wid];
      auto& interblock_edge_count_s_row_new = _pt_interblock_edge_count_s_row_new[wid];
      auto& interblock_edge_count_r_col_new = _pt_interblock_edge_count_r_col_new[wid];
      auto& interblock_edge_count_s_col_new = _pt_interblock_edge_count_s_col_new[wid];
      auto& block_degrees_out_new = _pt_block_degrees_out_new[wid];
      auto& block_degrees_in_new = _pt_block_degrees_in_new[wid];
      auto& block_degrees_new = _pt_block_degrees_new[wid];
      auto& r_row = _pt_r_row[wid];
      auto& s_row = _pt_s_row[wid];
      auto& r_col = _pt_r_col[wid];
      auto& s_col = _pt_s_col[wid];

      for (size_t proposal_idx = 0; 
        proposal_idx < num_agg_proposals_per_block; 
        proposal_idx++) {
        
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
          proposal,
          num_out_neighbor_edges,
          num_in_neighbor_edges,
          num_neighbor_edges,
          prob
        );
        
        _compute_new_rows_cols_interblock_edge_count_block(
          current_block,
          proposal,
          interblock_edge_count_r_row_new,
          interblock_edge_count_s_row_new,
          interblock_edge_count_r_col_new,
          interblock_edge_count_s_col_new
        );

        _compute_new_block_degree(
          current_block,
          proposal,
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
          interblock_edge_count_r_row_new,
          interblock_edge_count_s_row_new,
          interblock_edge_count_r_col_new,
          interblock_edge_count_s_col_new,
          block_degrees_out_new,
          block_degrees_in_new,
          r_row,
          s_row,
          r_col,
          s_col
        );

        if (delta_entropy < delta_entropy_for_each_block[current_block]) {
          best_merge_for_each_block[current_block] = proposal;
          delta_entropy_for_each_block[current_block] = delta_entropy;
        }
      }
  }).name("block_merge_par");

  tf::Task merging_update = _taskflow.emplace([this,
    &delta_entropy_for_each_block,
    &best_merge_for_each_block,
    &block_map] () {
      _argsort(delta_entropy_for_each_block);
      _carry_out_best_merges(
        best_merge_for_each_block,
        block_map
      );
      _initialize_edge_counts();    
  }).name("merging_update");

  tf::Task nodal_update_init = _taskflow.emplace([this, &itr_delta_entropy,
    &total_num_nodal_moves, &itr] () {
      itr_delta_entropy.clear();
      itr_delta_entropy.resize(max_num_nodal_itr, 0.0);
      total_num_nodal_moves = 0;
      itr = 0;
  }).name("nodal_update_init");

  tf::Task itr_init = _taskflow.emplace([this] () {
    std::fill(_pt_num_nodal_move_itr.begin(), _pt_num_nodal_move_itr.end(), 0);
    std::fill(_pt_delta_entropy_itr.begin(), _pt_delta_entropy_itr.end(), 0);
  }).name("itr_init");

  tf::Task nodal_update_par = _taskflow.for_each_index(0, int(_N), 1, 
    [this] (int current_node) {
      auto wid = _executor.this_worker_id();
      auto& prob = _pt_probabilities[wid];
      auto& neighbors = _pt_neighbors[wid];
      auto& interblock_edge_count_r_row_new = _pt_interblock_edge_count_r_row_new[wid];
      auto& interblock_edge_count_s_row_new = _pt_interblock_edge_count_s_row_new[wid];
      auto& interblock_edge_count_r_col_new = _pt_interblock_edge_count_r_col_new[wid];
      auto& interblock_edge_count_s_col_new = _pt_interblock_edge_count_s_col_new[wid];
      auto& block_degrees_out_new = _pt_block_degrees_out_new[wid];
      auto& block_degrees_in_new = _pt_block_degrees_in_new[wid];
      auto& block_degrees_new = _pt_block_degrees_new[wid];
      auto& r_row = _pt_r_row[wid];
      auto& s_row = _pt_s_row[wid];
      auto& r_col = _pt_r_col[wid];
      auto& s_col = _pt_s_col[wid];
      auto& num_nodal_move = _pt_num_nodal_move_itr[wid];
      auto& delta_entropy_itr = _pt_delta_entropy_itr[wid];

      size_t current_block = _partitions[current_node];

      size_t proposal;
      W num_out_neighbor_edges;
      W num_in_neighbor_edges;
      W num_neighbor_edges;
      _propose_new_partition_nodal(
        current_block,
        current_node,
        proposal,
        num_out_neighbor_edges,
        num_in_neighbor_edges,
        num_neighbor_edges,
        neighbors,
        prob
      );

      if (proposal != current_block) {
        _compute_new_rows_cols_interblock_edge_count_nodal(
          current_block,
          proposal,
          current_node,
          interblock_edge_count_r_row_new,
          interblock_edge_count_s_row_new,
          interblock_edge_count_r_col_new,
          interblock_edge_count_s_col_new
        );
        
        _compute_new_block_degree(
          current_block,
          proposal,
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
            interblock_edge_count_r_row_new,
            interblock_edge_count_r_col_new,
            block_degrees_new,
            r_row,
            r_col
          );
        }

        float delta_entropy = _compute_delta_entropy(
          current_block,
          proposal,
          interblock_edge_count_r_row_new,
          interblock_edge_count_s_row_new,
          interblock_edge_count_r_col_new,
          interblock_edge_count_s_col_new,
          block_degrees_out_new,
          block_degrees_in_new,
          r_row,
          s_row,
          r_col,
          s_col
        );
        
        float p_accept = std::min(
          static_cast<float>(std::exp(-beta * delta_entropy)) * Hastings_correction,
          1.0f
        );

        std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
        float rand_num = uni_dist(_generator);
        if ( rand_num <= p_accept) {
          num_nodal_move++;
          delta_entropy_itr += delta_entropy;
          _partitions[current_node] = proposal;
        }
      }
  }).name("nodal_update_par");
  
  tf::Task nodal_update = _taskflow.emplace([this, &itr_delta_entropy, itr] () {
    itr_delta_entropy[itr] = std::reduce(_pt_delta_entropy_itr.begin(), 
      _pt_delta_entropy_itr.end(), 0.0, [](float a, float b) { return a + b; });
    _initialize_edge_counts();    
  }).name("nodal_update");

  tf::Task check_mcmc = _taskflow.emplace([this,
    &itr, &interblock_edge_count_r_row_new, &itr_delta_entropy] () {
      float overall_entropy = _compute_overall_entropy(interblock_edge_count_r_row_new);
      if (itr >= (delta_entropy_moving_avg_window - 1)) {
        bool isfinite = true;
        isfinite = isfinite && std::isfinite(_old.overall_entropy_large);
        isfinite = isfinite && std::isfinite(_old.overall_entropy_med);
        isfinite = isfinite && std::isfinite(_old.overall_entropy_small);
        float mean = 0;
        std::reduce(itr_delta_entropy.begin() + (itr - delta_entropy_moving_avg_window + 1),
          itr_delta_entropy.end(), mean, [] (float a, float b) { return a + b; });
        mean /= (float)(delta_entropy_moving_avg_window - 1); 
        if (!isfinite) {
          if (-mean < (delta_entropy_threshold1 * overall_entropy)) {
            return 1;
          }
        }
        else {
          if (-mean < (delta_entropy_threshold2 * overall_entropy)) {
            return 1;
          }
        }
      }
      itr++;
      if (itr < max_num_nodal_itr) {
        return 0;
      } 
      else {
        return 1;
      }
  }).name("check_mcmc");

  tf::Task prepare_next = _taskflow.emplace([this, 
    &interblock_edge_count_r_row_new,
    &optimal_num_blocks_found] () {
      float overall_entropy = _compute_overall_entropy(interblock_edge_count_r_row_new);
      optimal_num_blocks_found = _prepare_for_partition_next(
        overall_entropy,
        num_block_reduction_rate
      );
      std::cout << optimal_num_blocks_found << std::endl;
      printf("Overall entropy: [%f, %f, %f] \n",
        _old.overall_entropy_large, _old.overall_entropy_med, _old.overall_entropy_small);
      printf("Number of blocks: [%ld, %ld, %ld] \n",
        _old.num_blocks_large, _old.num_blocks_med, _old.num_blocks_small);
  }).name("prepare_next");

  tf::Task find_opt = _taskflow.emplace([optimal_num_blocks_found] () {
    return optimal_num_blocks_found;
  }).name("find_opt");

  tf::Task finish = _taskflow.emplace([] () {}).name("finish");

  init.precede(block_merge_init);
  block_merge_init.precede(block_merge_par);
  block_merge_par.precede(merging_update);
  merging_update.precede(nodal_update_init);
  nodal_update_init.precede(itr_init);
  itr_init.precede(nodal_update_par);
  nodal_update_par.precede(nodal_update);
  nodal_update.precede(check_mcmc);
  check_mcmc.precede(itr_init, prepare_next);
  prepare_next.precede(find_opt);
  find_opt.precede(block_merge_init, finish);

  _executor.run(_taskflow).wait();
  _taskflow.dump(std::cout);
}

template <typename W>
void Graph_P<W>::_initialize_edge_counts() 
{

  _M.clear();
  _Mrow.clear();
  _Mcol.clear();

  _block_degrees_out.clear();
  _block_degrees_in.clear();
  _block_degrees.clear();
  _block_degrees_out.resize(_num_blocks);
  _block_degrees_in.resize(_num_blocks);
  _block_degrees.resize(_num_blocks);
  
  if (_num_blocks < 10000) {
    _M.resize(_num_blocks * _num_blocks, 0);
    for (size_t node = 0; node < _out_neighbors.size(); node++) {
      if (_out_neighbors[node].size() > 0) { 
        size_t k1 = _partitions[node];
        for (const auto& [v, w] : _out_neighbors[node]) {
          size_t k2 = _partitions[v];
          _M[k1*_num_blocks + k2] += w;
        }    
      }    
    }
    for (size_t i = 0; i < _num_blocks; i++) {
      for (size_t j = 0; j < _num_blocks; j++) {
        _block_degrees_out[i] += _M[i*_num_blocks + j];
        _block_degrees_in[i] += _M[j*_num_blocks + i];
      }
      _block_degrees[i] = _block_degrees_out[i] + _block_degrees_in[i];
    }
  }
  else {
    _Mrow.resize(_num_blocks);
    _Mcol.resize(_num_blocks);
    for (size_t node = 0; node < _out_neighbors.size(); node++) {
      if (_out_neighbors[node].size() > 0) { 
        size_t k1 = _partitions[node];
        for (const auto& [v, w] : _out_neighbors[node]) {
          size_t k2 = _partitions[v];
          _Mrow[k1].emplace_back(k2, w);
          _Mcol[k2].emplace_back(k1, w);
        }    
      }    
    }
    for (size_t i = 0; i < _num_blocks; i++) {
      for (const auto& [v, w] : _Mrow[i]) {
        _block_degrees_out[i] += w;
      }    
      for (const auto& [v, w] : _Mcol[i]) {
        _block_degrees_in[i] += w;
      }    
      _block_degrees[i] = _block_degrees_out[i] + _block_degrees_in[i];
    }
  }

} // end of initialize_edge_counts

template <typename W>
void Graph_P<W>::_propose_new_partition_block(
  size_t r,
  const std::vector<size_t>& partitions,
  size_t& s,
  W& k_out,
  W& k_in,
  W& k,
  std::vector<float>& prob)
{

  k_out = 0;
  k_in = 0;
  prob.resize(_num_blocks);
  
  if (_num_blocks < 10000) {
    for (size_t i = 0; i < _num_blocks; i++) {
      if (_M[_num_blocks*r + i] != 0) {
        k_out += _M[_num_blocks*r + i];
        prob[i] += (float)_M[_num_blocks*r + i];
      } 
      if (_M[_num_blocks*i + r] != 0) {
        k_in += _M[_num_blocks*i + r];
        prob[i] += (float)_M[_num_blocks*i + r];
      }
    }
  }
  else {
    for (const auto& [v, w] : _Mrow[r]) {
      k_out += w;
      prob[v] += w;
    }
    for (const auto& [v, w] : _Mcol[r]) {
      k_in += w;
      prob[v] += w;
    }
  }

  k = k_out + k_in;
  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if ( k == 0) {
    s = randint(const_cast<std::default_random_engine&>(_generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(), 
    [k](float p){ return p/(float)k; }
  );
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = dist(const_cast<std::default_random_engine&>(_generator));
  size_t u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(_generator));
  if ( x <= (float)_num_blocks/(_block_degrees[u]+_num_blocks) ) {
    std::uniform_int_distribution<int> choice(0, _num_blocks-1);
    int randIndex = choice(const_cast<std::default_random_engine&>(_generator));
    if (randIndex == r) randIndex++;
    if (randIndex == _num_blocks) randIndex = 0;
    s = randIndex;
  }
  else {
    prob.clear();
    prob.resize(_num_blocks);
    if (_num_blocks < 10000) {
      for (size_t i = 0; i < _num_blocks; i++) {
        prob[i] = (float)(_M[u*_num_blocks + i] + _M[i*_num_blocks + u])/_block_degrees[u];
      }
    }
    else {
      for (const auto& [v, w] : _Mrow[u]) {
        prob[v] += w;
      }
      for (const auto& [v, w] : _Mcol[u]) {
        prob[v] += w;
      }
      std::transform(prob.begin(), prob.end(), prob.begin(), 
        [this, u](float p){ 
          return p/(float)_block_degrees[u]; 
        }
      );
    }
    float multinomial_prob_sum = std::reduce(prob.begin(), prob.end(), 0.0);
    multinomial_prob_sum -= prob[r];
    prob[r] = 0;
    if (multinomial_prob_sum == 0) {
      std::uniform_int_distribution<int> choice(0, _num_blocks-1);
      int randIndex = choice(const_cast<std::default_random_engine&>(_generator));
      if (randIndex == r) randIndex++;
      if (randIndex == _num_blocks) randIndex = 0; 
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
    s = multinomial(const_cast<std::default_random_engine&>(_generator));
  }
}


template <typename W>
void Graph_P<W>::_propose_new_partition_nodal(
  size_t r,
  size_t ni,
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
    s = randint(const_cast<std::default_random_engine&>(_generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(), 
    [k](float p){
      return p/(float)k;
    }
  );
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(_generator))];
  size_t u = _partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(_generator));
  if ( x <= (float)_num_blocks/(_block_degrees[u]+_num_blocks) ) {
    s = randint(const_cast<std::default_random_engine&>(_generator));
  }
  else {
    prob.clear();
    prob.resize(_num_blocks);
    if (_num_blocks < 10000) {
      for (size_t i = 0; i < _num_blocks; i++) {
        prob[i] = (float)(_M[u*_num_blocks + i] + _M[i*_num_blocks + u])/_block_degrees[u];
      }
    }
    else {
      for (const auto& [v, w] : _Mrow[u]) {
        prob[v] += w;
      } 
      for (const auto& [v, w] : _Mcol[u]) {
        prob[v] += w;
      } 
      std::transform(prob.begin(), prob.end(), prob.begin(),
        [this, u](float p){
          return p/(float)_block_degrees[u];
        }
      );
    }
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(_generator));
  }
}

template <typename W>
void Graph_P<W>::_compute_new_rows_cols_interblock_edge_count_block(
  size_t r,
  size_t s,
  std::vector<W>& M_r_row,
  std::vector<W>& M_s_row,
  std::vector<W>& M_r_col,
  std::vector<W>& M_s_col)
{
  M_r_row.clear();
  M_r_col.clear();
  M_s_row.clear();
  M_s_col.clear();
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks);
  W count_in_sum_s = 0;
  W count_out_sum_s = 0;
  W count_self = 0;

  if (_num_blocks < 10000) {
    for (size_t i = 0; i < _num_blocks; i++) {
      M_s_row[i] = _M[s*_num_blocks + i];
      M_s_col[i] = _M[i*_num_blocks + s]; 
      if (_M[_num_blocks*r + i] != 0) {
        if (i == s) {
          count_out_sum_s += _M[_num_blocks*r + i];
        }
        if (i == r) {
          count_self += _M[_num_blocks*r + i]; 
        }
        M_s_row[i] += _M[_num_blocks*r + i];
      }
      if (_M[_num_blocks*i + r] != 0) {
        if (i == s) {
          count_in_sum_s += _M[_num_blocks*i + r];
        }
        M_s_col[i] += _M[_num_blocks*i + r];
      }
    } 
  }
  else {
    for (const auto& [v, w] : _Mrow[s]) {
      M_s_row[v] += w;
    }
    for (const auto& [v, w] : _Mcol[s]) {
      M_s_col[v] += w;
    }
    for (const auto& [v, w] : _Mrow[r]) {
      if (v == s) {
        count_out_sum_s += w;
      }
      if (v == r) {
        count_self += w;
      }
      M_s_row[v] += w;
    }
    for (const auto& [v, w] : _Mcol[r]) {
      if (v == s) {
        count_in_sum_s += w;
      }
      M_s_col[v] += w;
    }
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
  size_t r,
  size_t s,
  size_t ni,
  std::vector<W>& M_r_row,
  std::vector<W>& M_s_row,
  std::vector<W>& M_r_col,
  std::vector<W>& M_s_col) 
{
  M_r_row.clear();
  M_r_col.clear();
  M_s_row.clear();
  M_s_col.clear(); 
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks); 
  W count_out_sum_r = 0;
  W count_in_sum_r = 0;
  W count_out_sum_s = 0;
  W count_in_sum_s = 0;
  W count_self = 0;

  if (_num_blocks < 10000) {
    for (size_t i = 0; i < _num_blocks; i++) {
      M_r_row[i] = _M[r*_num_blocks + i];
      M_r_col[i] = _M[i*_num_blocks + r];
      M_s_row[i] = _M[s*_num_blocks + i];
      M_s_col[i] = _M[i*_num_blocks + s];
    } 
  }
  else {
    for (const auto& [v, w] : _Mrow[r]) {
      M_r_row[v] += w;
    }
    for (const auto& [v, w] : _Mcol[r]) {
      M_r_col[v] += w;
    }
    for (const auto& [v, w] : _Mrow[s]) {
      M_s_row[v] += w;
    }
    for (const auto& [v, w] : _Mcol[s]) {
      M_s_col[v] += w;
    }
  }
    
  size_t b;
  for (const auto& [v, w] : _out_neighbors[ni]) {
    b = _partitions[v];
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
    b = _partitions[v];
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
  W k_out,
  W k_in,
  W k,
  std::vector<W>& d_out_new,
  std::vector<W>& d_in_new,
  std::vector<W>& d_new) 
{

  d_out_new = _block_degrees_out;
  d_in_new = _block_degrees_in;
  d_new = _block_degrees;

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
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_s_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& M_s_col,
  const std::vector<W>& d_out_new,
  const std::vector<W>& d_in_new,
  std::vector<W>& r_row,
  std::vector<W>& s_row,
  std::vector<W>& r_col,
  std::vector<W>& s_col
  )
{
  
  // TODO: log, multiplication, division, are VERY expensive...
  // IADD / ISUB (integer point add and subtract) typically take 1 cycle
  // FADD / FSUB (floating point add and subtract) both take 3 cycles
  // FMUL (multiply) takes 5 cycles
  // FDIV (divide) takes 10-24 cycles
  // FYL2X (𝑦⋅log2(𝑥)) takes 90-106 cycles
  // F2XM1 (2𝑥−1) takes about 68 cycles
  // 
  // maybe a mixed strategy ... (if # iterations >= 10000, do parallel reduction)
  float delta_entropy = 0;

  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_row[i] != 0) {
      delta_entropy -= M_r_row[i] * std::log(static_cast<float>
        (M_r_row[i]) / (d_in_new[i] * d_out_new[r]));
    }
    if (M_s_row[i] != 0) {
      delta_entropy -= M_s_row[i] * std::log(static_cast<float>
        (M_s_row[i]) / (d_in_new[i] * d_out_new[s]));
    }
    // avoid duplicate counting
    if (i != r && i != s) {
      if (M_r_col[i] != 0) {
        delta_entropy -= M_r_col[i] * std::log(static_cast<float>
          (M_r_col[i]) / (d_out_new[i] * d_in_new[r]));
      }
      if (M_s_col[i] != 0) {
        delta_entropy -= M_s_col[i] * std::log(static_cast<float>
          (M_s_col[i]) / (d_out_new[i] * d_in_new[s]));
      }
    }
  }


  if (_num_blocks < 10000) {
    for (size_t i = 0; i < _num_blocks; i++) {
      if (_M[r*_num_blocks + i] != 0) {
        delta_entropy += _M[r*_num_blocks + i] * std::log(static_cast<float>
          (_M[r*_num_blocks + i]) / (_block_degrees_in[i] * _block_degrees_out[r]));
      }
      if (_M[s*_num_blocks + i] != 0) {
        delta_entropy += _M[s*_num_blocks + i] * std::log(static_cast<float>
          (_M[s*_num_blocks + i]) / (_block_degrees_in[i] * _block_degrees_out[s]));
      }
      // avoid duplicate counting
      if (i != r && i != s) {
        if (_M[i*_num_blocks + r] != 0) {
          delta_entropy += _M[i*_num_blocks + r] * std::log(static_cast<float>
            (_M[i*_num_blocks + r]) / (_block_degrees_out[i] * _block_degrees_in[r]));
        }
        if (_M[i*_num_blocks + s] != 0) {
          delta_entropy += _M[i*_num_blocks + s] * std::log(static_cast<float>
            (_M[i*_num_blocks + s]) / (_block_degrees_out[i] * _block_degrees_in[s]));
        }
      }
    } 
  }
  else {
    //remapping process
    r_row.clear();
    s_row.clear();
    r_col.clear();
    s_col.clear();
    r_row.resize(_num_blocks);
    s_row.resize(_num_blocks);
    r_col.resize(_num_blocks);
    s_col.resize(_num_blocks);
    for (const auto& [v, w] : _Mrow[r]) {
      if (w != 0) {   // TODO: do I need this??
        r_row[v] += w;
      }
    }
    for (const auto& [v, w] : _Mrow[s]) {
      if (w != 0) {
        s_row[v] += w;
      }
    }
    for (const auto& [v, w] : _Mcol[r]) {
      if (w != 0) {
        r_col[v] += w;
      }
    }
    for (const auto& [v, w] : _Mcol[s]) {
      if (w != 0) {
        s_col[v] += w;
      }
    }
    for (size_t v = 0; v < _num_blocks; v++) {
      if (r_row[v] != 0) {
        delta_entropy += r_row[v] * std::log(static_cast<float> (r_row[v]) / (_block_degrees_in[v] * _block_degrees_out[r]));
      }
      if (s_row[v] != 0) {
        delta_entropy += s_row[v] * std::log(static_cast<float> (s_row[v]) / (_block_degrees_in[v] * _block_degrees_out[s]));
      }
      if (v != r && v != s) {
        if (r_col[v] != 0) {
          delta_entropy += r_col[v] * std::log(static_cast<float> (r_col[v]) / (_block_degrees_out[v] * _block_degrees_in[r]));
        }
        if (s_col[v] != 0) {
          delta_entropy +=  s_col[v] * std::log(static_cast<float> (s_col[v]) / (_block_degrees_out[v] * _block_degrees_in[s]));
        }
      }
    }
  }
  
  return delta_entropy;
} // end of compute_delta_entropy


template <typename W>
void Graph_P<W>::_carry_out_best_merges(
  const std::vector<size_t>& best_merge_for_each_block,
  std::vector<size_t>& block_map)
{
  block_map.clear();
  block_map.resize(_num_blocks);
  std::iota(block_map.begin(), block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;

  while (num_merge < _num_blocks_to_merge) {
    int mergeFrom = _bestMerges[counter];
    int mergeTo = block_map[best_merge_for_each_block[_bestMerges[counter]]];
    counter++;
    if (mergeTo != mergeFrom) {
      for (size_t i = 0; i < _num_blocks; i++) {
        if (block_map[i] == mergeFrom) block_map[i] = mergeTo;
      }
      for (size_t i = 0; i < _partitions.size(); i++) {
        if (_partitions[i] == mergeFrom) _partitions[i] = mergeTo;
      }
      num_merge += 1;
    }
  }
  
  _unique(_partitions);
  block_map.clear();
  block_map.resize(_num_blocks, -1);
  for (size_t i = 0; i < _remaining_blocks.size(); i++) {
    block_map[_remaining_blocks[i]] = i;
  }

  for (auto& it : _partitions) {
    it = block_map[it];
  }
  _num_blocks = _num_blocks - _num_blocks_to_merge;
} // end of carry_out_best_merges

template <typename W>
float Graph_P<W>::_compute_overall_entropy(
  std::vector<W>& M_r_row)
{

  float data_S = 0;
  if (_num_blocks < 10000) { 
    for (size_t i = 0; i < _num_blocks; i++) { 
      for (size_t j = 0; j < _num_blocks; j++) {
        if (_M[i*_num_blocks + j] != 0) {
          data_S -= _M[i*_num_blocks + j] * std::log(_M[i*_num_blocks + j] / 
            (float)(_block_degrees_out[i] * _block_degrees_in[j]));
        }
      }
    } 
  }
  else {
    M_r_row.clear();
    M_r_row.resize(_num_blocks);
    for (size_t i = 0; i < _num_blocks; i++) {
      std::fill(M_r_row.begin(), M_r_row.end(), 0);
      for (const auto& [v, w] : _Mrow[i]) {
        M_r_row[v] += w;
      }
      for (size_t v = 0; v < _num_blocks; v++) {
        if (M_r_row[v] != 0) {
          data_S -= M_r_row[v] * std::log(M_r_row[v] / (float)(_block_degrees_out[i] * _block_degrees_in[v]));
        }
      }
    } 
  }

  float model_S_term = (float)_num_blocks*_num_blocks/_E;
  float model_S = (float)(_E * (1 + model_S_term) * std::log(1 + model_S_term)) - 
                          (model_S_term * log(model_S_term)) + (_N * log(_num_blocks));

  return model_S + data_S;
} // end of compute_overall_entropy


template <typename W>
float Graph_P<W>::_compute_Hastings_correction(
  size_t s,
  size_t ni, 
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& d_new,
  std::vector<W>& Mrows,
  std::vector<W>& Mcols) 
{
  float p_forward = 0;
  float p_backward = 0;

  if (_num_blocks < 10000) { 
    size_t b;
    for (const auto& [v, w] : _out_neighbors[ni]) {
      b = _partitions[v];
      p_forward += (float)(w * (_M[b*_num_blocks + s] + _M[s*_num_blocks 
        + b] + 1)) / (_block_degrees[b] + _num_blocks);
      p_backward += (float)(w * (M_r_row[b] + M_r_col[b] + 1)) / 
        (d_new[b] + _num_blocks);
    }
    for (const auto& [v, w] : _in_neighbors[ni]) {
      b = _partitions[v];
      p_forward += (float)(w * (_M[b*_num_blocks + s] + _M[s*_num_blocks 
        + b] + 1)) / (_block_degrees[b] + _num_blocks);
      p_backward += (float)(w * (M_r_row[b] + M_r_col[b] + 1)) / 
        (d_new[b] + _num_blocks);
    } 
  }
  else {
    Mrows.clear();
    Mcols.clear();
    Mrows.resize(_num_blocks);
    Mcols.resize(_num_blocks);
    for (const auto& [v, w] : _Mrow[s]) {
      Mrows[v] += w;
    }
    for (const auto& [v, w] : _Mcol[s]) {
      Mcols[v] += w;
    }
    size_t b;
    for (const auto& [v, w] : _out_neighbors[ni]) {
      b = _partitions[v];
      p_forward += (float)w*(Mcols[b] + Mrows[b] + 1)/(_block_degrees[b]+_num_blocks);
      p_backward += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
    }
    for (const auto & [v, w] : _in_neighbors[ni]) {
      b = _partitions[v];
      p_forward += (float)w*(Mcols[b] + Mrows[b] + 1)/(_block_degrees[b]+_num_blocks);
      p_backward += (float)w*(M_r_row[b] + M_r_col[b] + 1)/(d_new[b]+_num_blocks);
    }
  }

  return p_backward / p_forward;
} // end of compute_Hastings_correction


template <typename W>
bool Graph_P<W>::_prepare_for_partition_next(
  float S,
  float B_rate) 
{
  
  bool optimal_B_found = false;
  int index;
  
  if (S <= _old.overall_entropy_med) { 
    if (_old.num_blocks_med > _num_blocks) { 
      _old.partitions_large =       _old.partitions_med;
      _old.M_large =                _old.M_med;
      _old.Mrow_large =             _old.Mrow_med;
      _old.Mcol_large =             _old.Mcol_med;   
      _old.block_degree_large =     _old.block_degree_med;
      _old.block_degree_in_large =  _old.block_degree_in_med;
      _old.block_degree_out_large = _old.block_degree_out_med;
      _old.overall_entropy_large =  _old.overall_entropy_med;
      _old.num_blocks_large =       _old.num_blocks_med;
    }
    else {
      _old.partitions_small =       _old.partitions_med;
      _old.M_small =                _old.M_med;
      _old.Mrow_small =             _old.Mrow_med;
      _old.Mcol_small =             _old.Mcol_med;
      _old.block_degree_small =     _old.block_degree_med;
      _old.block_degree_in_small =  _old.block_degree_in_med;
      _old.block_degree_out_small = _old.block_degree_out_med;
      _old.overall_entropy_small =  _old.overall_entropy_med;
      _old.num_blocks_small =       _old.num_blocks_med;  
    }
    _old.partitions_med =       _partitions; 
    _old.M_med =                _M;
    _old.Mrow_med =             _Mrow;
    _old.Mcol_med =             _Mcol;
    _old.block_degree_med =     _block_degrees;
    _old.block_degree_in_med =  _block_degrees_in;
    _old.block_degree_out_med = _block_degrees_out;
    _old.overall_entropy_med =  S;
    _old.num_blocks_med =       _num_blocks;
  }
  else {
    if (_old.num_blocks_med > _num_blocks) {
      _old.partitions_small =       _partitions;
      _old.M_small =                _M;
      _old.Mrow_small =             _Mrow;
      _old.Mcol_small =             _Mcol;
      _old.block_degree_small =     _block_degrees;
      _old.block_degree_in_small =  _block_degrees_in;
      _old.block_degree_out_small = _block_degrees_out;
      _old.overall_entropy_small =  S;
      _old.num_blocks_small =       _num_blocks; 
    }
    else {
      _old.partitions_large =       _partitions;
      _old.M_large =                _M;
      _old.Mrow_large =             _Mrow;
      _old.Mcol_large =             _Mcol;
      _old.block_degree_large =     _block_degrees;
      _old.block_degree_in_large =  _block_degrees_in;
      _old.block_degree_out_large = _block_degrees_out;
      _old.overall_entropy_large =  S;
      _old.num_blocks_large =       _num_blocks;
    }
  }
 
  if (std::isinf(_old.overall_entropy_small)) {
    _num_blocks_to_merge = (int)_num_blocks*B_rate;
    if (_num_blocks_to_merge == 0)  optimal_B_found = true;
    _partitions =         _old.partitions_med;
    _M =                  _old.M_med;
    _Mrow =               _old.Mrow_med;
    _Mcol =               _old.Mcol_med;
    _block_degrees =      _old.block_degree_med;
    _block_degrees_out =  _old.block_degree_out_med;
    _block_degrees_in =   _old.block_degree_in_med; 
  }
  else {
    if (_old.num_blocks_large - _old.num_blocks_small == 2) {
      optimal_B_found =   true;
      _num_blocks =       _old.num_blocks_med;
      _partitions =       _old.partitions_med;
    }
    else {
      if ((_old.num_blocks_large - _old.num_blocks_med) >= 
          (_old.num_blocks_med - _old.num_blocks_small)) {  
        int next_B_to_try = _old.num_blocks_med + 
          static_cast<int>(std::round((_old.num_blocks_large - _old.num_blocks_med) * 0.618));
        _num_blocks_to_merge = _old.num_blocks_large - next_B_to_try;
        _num_blocks =           _old.num_blocks_large;
        _partitions =           _old.partitions_large;
        _M =                    _old.M_large;
        _Mrow =                 _old.Mrow_large;
        _Mcol =                 _old.Mcol_large;
        _block_degrees =        _old.block_degree_large;
        _block_degrees_out =    _old.block_degree_out_large;
        _block_degrees_in =     _old.block_degree_in_large;
      }
      else {
        int next_B_to_try =   _old.num_blocks_small 
          + static_cast<int>(std::round((_old.num_blocks_med - _old.num_blocks_small) * 0.618));
        _num_blocks_to_merge = _old.num_blocks_med - next_B_to_try;
        _num_blocks =           _old.num_blocks_med;
        _partitions =           _old.partitions_med;
        _M =                    _old.M_med;
        _Mrow =                 _old.Mrow_med;
        _Mcol =                 _old.Mcol_med;
        _block_degrees =        _old.block_degree_med;
        _block_degrees_out =    _old.block_degree_out_med;
        _block_degrees_in =     _old.block_degree_in_med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


