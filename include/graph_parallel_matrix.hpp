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
#include <set>
#include <unordered_map>
#include <thread>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <cassert>

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
    int beta                           = 3;
    size_t num_agg_proposals_per_block = 10; 
    float num_block_reduction_rate     = 0.5;
    size_t max_num_nodal_itr           = 100;
    float delta_entropy_threshold1     = 5e-4;
    float delta_entropy_threshold2     = 1e-4;
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
      _pt_num_nodal_move_itr(num_threads),
      _pt_delta_entropy_itr(num_threads)
    {
      load_graph_from_tsv(FileName);
      _generator.seed(_rd());
    }

    // partition ground truth
    std::vector<size_t> truePartitions;

  private:

    size_t _N;  // number of node
    size_t _E;  // number of edge
    
    std::vector<Edge> _edges;
    std::vector<std::vector<std::pair<size_t, W>>> _out_neighbors;
    std::vector<std::vector<std::pair<size_t, W>>> _in_neighbors;

    size_t _num_blocks;

    std::random_device _rd;
    std::default_random_engine _generator;
   
    tf::Executor _executor;
    tf::Taskflow _taskflow;
          
    std::vector< std::vector<size_t>> _pt_neighbors;
    std::vector< std::vector<float>>  _pt_probabilities;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_r_row_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_s_row_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_r_col_new;
    std::vector< std::vector<W>>      _pt_interblock_edge_count_s_col_new;
    std::vector< std::vector<W>>      _pt_block_degrees_out_new;
    std::vector< std::vector<W>>      _pt_block_degrees_in_new;
    std::vector< std::vector<W>>      _pt_block_degrees_new;
    std::vector< int >                _pt_num_nodal_move_itr;
    std::vector< float >              _pt_delta_entropy_itr;

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

    Old _old;
    size_t _num_blocks;
    std::vector<size_t> _partitions;
    std::vector<W> _M;
    std::vector<W> _d_out;
    std::vector<W> _d_in;
    std::vector<W> _d;
    std::vector<size_t> _bestMerges;
    std::vector<size_t> _remaining_blocks;
    std::set<size_t> _seen;

    // functions used internally
    void _initialize_edge_counts();
 
    void _propose_new_partition_block(
      size_t r,
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
      const std::vector<size_t>& partitions,
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
      const std::vector<W>& d_in_new
    );        
     
    size_t _carry_out_best_merges(
      const std::vector<int>& best_merge_for_each_block,
      size_t B_to_merge,
      std::vector<size_t>& block_map,
    );

    float _compute_overall_entropy(); 

    float _compute_Hastings_correction(
      size_t s,
      size_t ni,
      const std::vector<W>& M_r_row,
      const std::vector<W>& M_r_col,
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
    );

    bool _prepare_for_partition_next(
      float S,
      size_t& B_to_merge,
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
      _remaining_blocks.clear();
      _seen.clear();
      for (const auto& elem : arr) {
        if (_seen.find(elem) == _seen.end()) {
          _remaining_blocks.emplace_back(elem);
          _seen.insert(elem);
        }   
      }
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
std::vector<size_t> Graph_P<W>::partition() {
  
  _num_blocks = _N;
  _partitions.clear();
  _partitions.resize(_num_blocks);
  std::iota(_partitions.begin(), _partitions.end(), 0);
 
  _M.clear();
  _d_out.clear();
  _d_in.clear();
  _d.clear();

  _initialize_edge_counts();

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
  std::vector<size_t> block_map;
  std::vector<size_t> block_partition;

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

    size_t start_block = 0;
    size_t step = 1;
    _taskflow.clear();
    _taskflow.for_each_index(std::ref(start_block), std::ref(_num_blocks), step, 
      [this,
      &block_partition,
      &best_merge_for_each_block,
      &delta_entropy_for_each_block
      ] (size_t current_block) {
      
          auto wid = _executor.this_worker_id();
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
    
    _argsort(delta_entropy_for_each_block);
    _num_blocks = _carry_out_best_merges(
      best_merge_for_each_block,
      num_blocks_to_merge,
      block_map,
    );
   
    _M.clear(); 
    _d_out.clear();
    _d_in.clear();
    _d.clear();
    _initialize_edge_counts();

    int total_num_nodal_moves = 0;
    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);

    float overall_entropy = _compute_overall_entropy();

    if (verbose)
      printf("overall_entropy: %f\n", overall_entropy);

    // nodal updates
    for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {

      int num_nodal_moves = 0;
      std::fill(_pt_num_nodal_move_itr.begin(), _pt_num_nodal_move_itr.end(), 0);
      std::fill(_pt_delta_entropy_itr.begin(), _pt_delta_entropy_itr.end(), 0);

      size_t start_node = 0;
      size_t step = 1;
      _taskflow.clear();
      _taskflow.for_each_index(std::ref(start_block), std::ref(_N), step,
        [this] (size_t current_node) {

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
          
            interblock_edge_count_r_row_new.clear();
            interblock_edge_count_s_row_new.clear();
            interblock_edge_count_r_col_new.clear();
            interblock_edge_count_s_col_new.clear();
            _compute_new_rows_cols_interblock_edge_count_nodal(
              current_block,
              proposal,
              current_node,
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
                block_degrees_new
              );
            } // calculate Hastings_correction
            
            float delta_entropy = _compute_delta_entropy(
              current_block,
              proposal,
              interblock_edge_count_r_row_new,
              interblock_edge_count_s_row_new,
              interblock_edge_count_r_col_new,
              interblock_edge_count_s_col_new,
              block_degrees_out_new,
              block_degrees_in_new
            );

            float p_accept = std::min(static_cast<float>(std::exp(-beta * delta_entropy))
                                                         * Hastings_correction, 1.0f);


            std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
            float rand_num = uni_dist(_generator);
            if ( rand_num <= p_accept) {
              num_nodal_move++;
              delta_entropy_itr += delta_entropy;
              _partitions[current_node] = proposal;
            }
        } // end if 
      } // end current_node
      );
      _executor.run(_taskflow).wait();

      num_nodal_moves = std::reduce(_pt_num_nodal_move_itr.begin(), _pt_num_nodal_move_itr.end(), 0,
        [](int a, int b) { return a + b; }
      );
      itr_delta_entropy[itr] = std::reduce(_pt_delta_entropy_itr.begin(), _pt_delta_entropy_itr.end(), 0.0,
        [](float a, float b) { return a + b; }
      );
      total_num_nodal_moves += num_nodal_moves;

      _initialize_edge_counts();

      float oe = overall_entropy = _compute_overall_entropy();

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
    
    overall_entropy = _compute_overall_entropy();

    if (verbose)
      printf("Total number of nodal moves: %d, overall_entropy: %.5f\n", 
              total_num_nodal_moves, overall_entropy);


    optimal_num_blocks_found = _prepare_for_partition_next(
      overall_entropy,
      num_blocks_to_merge,
      num_block_reduction_rate
    );

    if (verbose) {
      printf("Overall entropy: [%f, %f, %f] \n", 
        _old.overall_entropy_large, _old.overall_entropy_med, _old.overall_entropy_small);
      printf("Number of blocks: [%ld, %ld, %ld] \n",
        _old.num_blocks_large, _old.num_blocks_med, _old.num_blocks_small);
    }
  } // end while

  return partitions;
}

template <typename W>
void Graph_P<W>::_initialize_edge_counts()
{

  _taskflow.clear();
  _M.clear();
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

  // compute initial block degrees
  _d_out.clear();
  _d_out.resize(_num_blocks);
  _d_in.clear();
  _d_in.resize(_num_blocks);
  _d.clear();
  _d.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    _taskflow.emplace([i, this] () {
      for (size_t j = 0; j < _num_blocks; j++) {
        _d_out[i] += _M[i*_num_blocks + j]; // row
        _d_in[i] += _M[j*_num_blocks + i]; // col
      }
      _d[i] = _d_out[i] + _d_in[i];
    });
  }
  _executor.run(_taskflow).wait();

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

  k = k_out + k_in;
  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if ( k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  //for (auto& p : prob) p/=k;
  std::transform(prob.begin(), prob.end(), prob.begin(),
    [&k](float i) { return (float) i/k; });
  
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = dist(const_cast<std::default_random_engine&>(generator));
  size_t u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(_generator));
  if ( x <= (float)_num_blocks/(_d[u]+_num_blocks) ) {
    std::uniform_int_distribution<int> choice(0, _num_blocks-1);
    int randIndex = choice(const_cast<std::default_random_engine&>(_generator));
    if (randIndex == r) randIndex++;
    if (randIndex == _num_blocks) randIndex = 0;
    s = randIndex;
  }
  else {
    prob.clear();
    prob.resize(_num_blocks);
    float multinomial_prob_sum = 0;
    for (size_t i = 0; i < _num_blocks; i++) {
      prob[i] = (float)(_M[u*_num_blocks + i] + _M[i*_num_blocks + u])/_d[u];
      multinomial_prob_sum += prob[i];
    }
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
      for (auto& it : prob) it /= multinomial_prob_sum;
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
    prob.emplace_back(w);
    k_out += w;
  }
  for (const auto& [v, w] : _in_neighbors[ni]) {
    neighbors.emplace_back(v);
    prob.emplace_back(w);
    k_in += w;
  }
  
  k = k_out + k_in;
  std::uniform_int_distribution<int> randint(0, _num_blocks-1);
  if (k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(),
    [&k](float i) { return (float) i/k; }); 
  
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
    M_s_row[i] = _M[s*_num_blocks + i];
    M_s_col[i] = _M[i*_num_blocks + s]; 
    if (_M[_num_blocks*r + i] != 0) {
      if (i == s) {
        count_out_sum_s += _M[_num_blocks*r + i];
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

  M_s_row[r] -= count_in_sum_s;
  M_s_row[s] += count_in_sum_s;
  M_s_row[r] -= _M[r*_num_blocks + r];
  M_s_row[s] += _M[r*_num_blocks + r];
  
  M_s_col[r] -= count_out_sum_s;
  M_s_col[s] += count_out_sum_s;
  M_s_col[r] -= _M[r*_num_blocks + r];
  M_s_col[s] += _M[r*_num_blocks + r];
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
    M_r_row[i] = _M[r*_num_blocks + i];
    M_r_col[i] = _M[i*_num_blocks + r];
    M_s_row[i] = _M[s*_num_blocks + i];
    M_s_col[i] = _M[i*_num_blocks + s];
  } 
  // out_blocks
  for (const auto& [v, w] : _out_neighbors[ni]) {
    if (_partitions[v] == r) {
      count_out_sum_r += w;
    }
    if (partitions[v] == s) {
      count_out_sum_s += w;
    }
    if (v == ni) {
      count_self += w;
    }
    M_r_row[ _partitions[v] ] -= w;
    M_s_row[ _partitions[v] ] += w;
  }
  // in_blocks
  for (const auto& [v, w] : _in_neighbors[ni]) {
    if (_partitions[v] == r) {
      count_in_sum_r += w;
    }
    if (_partitions[v] == s) {
      count_in_sum_s += w;
    }
    M_r_col[ _partitions[v] ] -= w;
    M_s_col[ _partitions[v] ] += w;
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

  d_out_new = _d_out;
  d_in_new = _d_in;
  d_new = _d;

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
  const std::vector<W>& d_in_new) 
{
  
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
    if (_M[r*_num_blocks + i] != 0) {
      delta_entropy += _M[r*_num_blocks + i] * std::log(static_cast<float>
        (_M[r*_num_blocks + i]) / (_d_in[i] * _d_out[r]));
    }
    if (_M[s*_num_blocks + i] != 0) {
      delta_entropy += _M[s*_num_blocks + i] * std::log(static_cast<float>
        (_M[s*_num_blocks + i]) / (_d_in[i] * _d_out[s]));
    }
    // avoid duplicate counting
    if (i != r && i != s) {
      if (_M[i*_num_blocks + r] != 0) {
        delta_entropy += _M[i*_num_blocks + r] * std::log(static_cast<float>
          (_M[i*_num_blocks + r]) / (_d_out[i] * _d_in[r]));
      }
      if (_M[i*_num_blocks + s] != 0) {
        delta_entropy += _M[i*_num_blocks + s] * std::log(static_cast<float>
          (_M[i*_num_blocks + s]) / (_d_out[i] * _d_in[s]));
      }
    }
  }
  
  return delta_entropy;
} // end of compute_delta_entropy


template <typename W>
size_t Graph_P<W>::_carry_out_best_merges(
  const std::vector<int>& best_merge_for_each_block,
  size_t B_to_merge,
  std::vector<size_t>& block_map) 
{
  block_map.clear();
  block_map.resize(_num_blocks);
  std::iota(block_map.begin(), block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;

  while (num_merge < B_to_merge) {
    int mergeFrom = _bestMerges[counter];
    int mergeTo = block_map[best_merge_for_each_block[_bestMerges[counter]]];
    counter++;
    if (mergeTo != mergeFrom) {
      for (size_t i = 0; i < _num_blocks; i++) {
        if (block_map[i] == mergeFrom) {
          block_map[i] = mergeTo;
        }
      }
      for (size_t i = 0; i < _partitions.size(); i++) {
        if (_partitions[i] == mergeFrom) {
          _partitions[i] = mergeTo;
        }
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

  for (auto& it : b) {
    it = block_map[it];
  }

  return B - B_to_merge;
} // end of carry_out_best_merges

template <typename W>
float Graph_P<W>::_compute_overall_entropy() 
{
  
  float data_S = 0;
  for (size_t i = 0; i < _num_blocks; i++) { 
    for (size_t j = 0; j < _num_blocks; j++) {
      if (_M[i*_num_blocks + j] != 0) {
        data_S -= M[i*_num_blocks + j] * std::log(_M[i*_num_blocks + j] / 
          (float)(_d_out[i] * _d_in[j]));
      }
    }
  }

  float model_S_term = (float)_num_blocks*_num_blocks/_E;
  float model_S = (float)(_E * (1 + model_S_term) * std::log(1 + model_S_term)) - 
                          (model_S_term * std::log(model_S_term)) + (_N * log(_num_blocks));
  
  return model_S + data_S;

} // end of compute_overall_entropy


template <typename W>
float Graph_P<W>::_compute_Hastings_correction(
  size_t s,
  size_t ni, 
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& d_new) 
{
  float p_forward = 0;
  float p_backward = 0;

  size_t block;
  for (const auto& [v, w] : _out_neighbors[ni]) {
    block = _partitions[v];
    p_forward += (float)(w * (_M[block*_num_blocks + s] + _M[s*_num_blocks 
      + block] + 1)) / (_d[block] + _num_blocks);
    p_backward += (float)(w * (M_r_row[block] + M_r_col[block] + 1)) / 
      (d_new[block] + _num_blocks);
  }
  
  for (const auto& [v, w] : _in_neighbors[ni]) {
    block = _partitions[v];
    p_forward += (float)(w * (_M[block*_num_blocks + s] + _M[s*_num_blocks 
      + block] + 1)) / (_d[block] + _num_blocks);
    p_backward += (float)(w * (M_r_row[block] + M_r_col[block] + 1)) / 
      (d_new[block] + _num_blocks);
  } 
  
  return p_backward / p_forward;
} // end of compute_Hastings_correction

template <typename W>
bool Graph_P<W>::_prepare_for_partition_next(
  float S,
  size_t& B_to_merge,
  float B_rate) 
{
  bool optimal_B_found = false;
  int index;
  
  if (S <= _old.overall_entropy_med) {  // if the current overall entropy is the best so far
    if (_old.num_blocks_med > _num_blocks) { 
      _old.partitions_large =            _old.partitions_med;
      _old.interblock_edge_count_large = _old.interblock_edge_count_med;
      _old.block_degree_large =          _old.block_degree_med;
      _old.block_degree_in_large =       _old.block_degree_in_med;
      _old.block_degree_out_large =      _old.block_degree_out_med;
      _old.overall_entropy_large =       _old.overall_entropy_med;
      _old.num_blocks_large =            _old.num_blocks_med;
    }
    else {
      _old.partitions_small =            _old.partitions_med;
      _old.interblock_edge_count_small = _old.interblock_edge_count_med;
      _old.block_degree_small =          _old.block_degree_med;
      _old.block_degree_in_small =       _old.block_degree_in_med;
      _old.block_degree_out_small =      _old.block_degree_out_med;
      _old.overall_entropy_small =       _old.overall_entropy_med;
      _old.num_blocks_small =            _old.num_blocks_med;  
    }
    _old.partitions_med            = _partitions; 
    _old.interblock_edge_count_med = _M;
    _old.block_degree_med          = _d;
    _old.block_degree_in_med       = _d_in;
    _old.block_degree_out_med      = _d_out;
    _old.overall_entropy_med       = S;
    _old.num_blocks_med            = _num_blocks;
  }
  else {
    if (_old.num_blocks_med > _num_blocks) {
      _old.partitions_small            = _partitions;
      _old.interblock_edge_count_small = _M;
      _old.block_degree_small          = _d;
      _old.block_degree_in_small       = _d_in;
      _old.block_degree_out_small      = _d_out;
      _old.overall_entropy_small       = S;
      _old.num_blocks_small            = _num_blocks; 
    }
    else {
      _old.partitions_large            = _partitions;
      _old.interblock_edge_count_large = _M;
      _old.block_degree_large          = _d;
      _old.block_degree_in_large       = _d_in;
      _old.block_degree_out_large      = _d_out;
      _old.overall_entropy_large       = S;
      _old.num_blocks_large            = _num_blocks;
    }
  }
 
  if (std::isinf(_old.overall_entropy_small)) {
    B_to_merge = (int)_num_blocks*B_rate;
    if (B_to_merge == 0) optimal_B_found = true;
    partitions = _old.partitions_med;
    _M         = _old.interblock_edge_count_med;
    _d         = _old.block_degree_med;
    _d_out     = _old.block_degree_out_med;
    _d_in      = _old.block_degree_in_med; 
  }
  else {
    if (_old.num_blocks_large - _old.num_blocks_small == 2) {
      optimal_B_found = true;
      _num_blocks = _old.num_blocks_med;
      partitions  = _old.partitions_med;
    }
    else {
      if ((_old.num_blocks_large - _old.num_blocks_med) >= 
        (_old.num_blocks_med - _old.num_blocks_small)) {  
        int next_B_to_try = _old.num_blocks_med + 
          static_cast<int>(std::round((_old.num_blocks_large - _old.num_blocks_med) * 0.618));
        B_to_merge  = _old.num_blocks_large - next_B_to_try;
        _num_blocks = _old.num_blocks_large;
        partitions  = _old.partitions_large;
        _M          = _old.interblock_edge_count_large;
        _d          = _old.block_degree_large;
        _d_out      = _old.block_degree_out_large;
        _d_in       = _old.block_degree_in_large;
      }
      else {
        int next_B_to_try = _old.num_blocks_small 
          + static_cast<int>(std::round((_old.num_blocks_med - _old.num_blocks_small) * 0.618));
        B_to_merge  = _old.num_blocks_med - next_B_to_try;
        _num_blocks = _old.num_blocks_med;
        _partitions = _old.partitions_med;
        _M          = _old.interblock_edge_count_med;
        _d          = _old.block_degree_med;
        _d_out      = _old.block_degree_out_med;
        _d_in       = _old.block_degree_in_med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


