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
    };
  
    Old _old;  
    std::vector< std::vector<std::pair<size_t, W>> > _Mrow;
    std::vector< std::vector<std::pair<size_t, W>> > _Mcol;
    std::vector<size_t> _partitions;
    std::vector<W> _block_degrees_out;
    std::vector<W> _block_degrees_in;
    std::vector<W> _block_degrees;


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
      const std::vector<W>& d_in_new
      //std::vector<W>& Mrows,
      //std::vector<W>& Mcols,
      //std::vector<W>& Mrows2,
      //std::vector<W>& Mcols2
    );        
     
    size_t _carry_out_best_merges(
      const std::vector<size_t>& bestMerges,
      const std::vector<int>& best_merge_for_each_block,
      size_t B_to_merge,
      std::vector<size_t>& block_map,
      std::vector<size_t>& remaining_blocks
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
      const std::vector<W>& d_new
    );

    bool _prepare_for_partition_next(
      float S,
      size_t& B_to_merge,
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
  
  _partitions.resize(_num_blocks);
  std::iota(_partitions.begin(), _partitions.end(), 0); 
  
  _Mrow.clear();
  _Mcol.clear();

  _block_degrees_out.clear();
  _block_degrees_in.clear();
  _block_degrees.clear();

  _initialize_edge_counts();

  _old.overall_entropy_large =  std::numeric_limits<float>::infinity();
  _old.overall_entropy_med =    std::numeric_limits<float>::infinity();
  _old.overall_entropy_small =  std::numeric_limits<float>::infinity();

  _old.num_blocks_large = 0;
  _old.num_blocks_med =   0;
  _old.num_blocks_small = 0;

  bool optimal_num_blocks_found = false;

  size_t num_blocks_to_merge = (size_t)_num_blocks * num_block_reduction_rate;

  std::vector<int> best_merge_for_each_block;
  std::vector<float> delta_entropy_for_each_block;
  std::vector<size_t> bestMerges;
  std::vector<size_t> block_map;
  std::vector<size_t> block_partition;

  // proposal_nodal
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

  //Hasting Correction
  //Delta Entropy
  std::vector<W> Mrows;
  std::vector<W> Mcols;
  std::vector<W> Mrows2;
  std::vector<W> Mcols2;

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
    
    // TODO: try to use for_each instead of explicitly creating B tasks
    _taskflow.clear();
    for (size_t current_block = 0; current_block < _num_blocks; current_block++) {
      _taskflow.emplace([this,
        &Mrows,
        &Mcols,
        &Mrows2,
        &Mcols2,
        &block_partition,
        &best_merge_for_each_block,
        &delta_entropy_for_each_block,
        current_block ](){
          
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

            interblock_edge_count_r_row_new.clear();
            interblock_edge_count_s_row_new.clear();
            interblock_edge_count_r_col_new.clear();
            interblock_edge_count_s_col_new.clear();
            _compute_new_rows_cols_interblock_edge_count_block(
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
              //Mrows,
              //Mcols,
              //Mrows2,
              //Mcols2
            );
            
            if (delta_entropy < delta_entropy_for_each_block[current_block]) {
              best_merge_for_each_block[current_block] = proposal;
              delta_entropy_for_each_block[current_block] = delta_entropy;
            }     
          } // end for proposal_idx
      });
    }
    
    _executor.run(_taskflow).wait();


    bestMerges = _argsort(delta_entropy_for_each_block);
    _num_blocks = _carry_out_best_merges(
      bestMerges,
      best_merge_for_each_block,
      num_blocks_to_merge,
      block_map,
      remaining_blocks
    );

    _Mrow.clear();
    _Mcol.clear();
    _block_degrees_out.clear();
    _block_degrees_in.clear();
    _block_degrees.clear();
    _initialize_edge_counts();

    int total_num_nodal_moves = 0;
    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);

    float overall_entropy = _compute_overall_entropy(interblock_edge_count_r_row_new);
    
    if (verbose)
      printf("overall_entropy: %f\n", overall_entropy);
 
    // nodal updates
    for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {

      int num_nodal_moves = 0;
      itr_delta_entropy[itr] = 0;

      for (size_t current_node = 0; current_node < _N; current_node++) {
      
        size_t current_block = _partitions[current_node];
     
        neighbors_nodal.clear();
        prob_nodal.clear(); 
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
          neighbors_nodal,
          prob_nodal
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
              block_degrees_new,
              Mrows,
              Mcols
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
            //Mrows,
            //Mcols,
            //Mrows2,
            //Mcols2
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
              block_degrees_new
            );
          }
        } // end if 
      } // end current_node

      float oe = overall_entropy = _compute_overall_entropy(interblock_edge_count_r_row_new);

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
    
    overall_entropy = _compute_overall_entropy(interblock_edge_count_r_row_new);

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
  return _partitions;
}

template <typename W>
void Graph_P<W>::_initialize_edge_counts() 
{

  //_taskflow.clear();
  _Mrow.resize(_num_blocks);
  _Mcol.resize(_num_blocks);
 
  for (size_t node = 0; node < _out_neighbors.size(); node++) {
    if (_out_neighbors[node].size() > 0) {
      size_t k1 = _partitions[node];
      for (const auto& [v, w] : _out_neighbors[node]) {
        size_t k2 = _partitions[v];
        _Mrow[k1].emplace_back(std::make_pair(k2, w));
        _Mcol[k2].emplace_back(std::make_pair(k1, w));
      }
    }
  }

  _block_degrees_out.resize(_num_blocks);
  _block_degrees_in.resize(_num_blocks);
  _block_degrees.resize(_num_blocks);
  for (size_t i = 0; i < _num_blocks; i++) {
    for (const auto& [v, w] : _Mrow[i]) {
      _block_degrees_out[i] += w;
    }
    for (const auto& [v, w] : _Mcol[i]) {
      _block_degrees_in[i] += w;
    }
    _block_degrees[i] = _block_degrees_out[i] + _block_degrees_in[i];
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
  for (const auto& [v, w] : _Mrow[r]) {
    k_out += w;
    prob[v] += w;
  }
  for (const auto& [v, w] : _Mcol[r]) {
    k_in += w;
    prob[v] += w;
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
    float multinomial_prob_sum = 0;
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
    multinomial_prob_sum = std::reduce(prob.begin(), prob.end(), 0.0);
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
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks);
  W count_in_sum_s = 0;
  W count_out_sum_s = 0;
  W count_self = 0;
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
 
  M_r_row.resize(_num_blocks);
  M_r_col.resize(_num_blocks);
  M_s_row.resize(_num_blocks);
  M_s_col.resize(_num_blocks); 
  W count_out_sum_r = 0;
  W count_in_sum_r = 0;
  W count_out_sum_s = 0;
  W count_in_sum_s = 0;
  W count_self = 0;
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
  const std::vector<W>& d_in_new
  //std::vector<W>& Mrows,
  //std::vector<W>& Mcols,
  //std::vector<W>& Mrows2,
  //std::vector<W>& Mcols2
  )
{
  
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
  //Mrows.clear();
  //Mcols.clear();
  //Mrows2.clear();
  //Mcols2.clear();
  //Mrows.resize(_num_blocks);
  //Mcols.resize(_num_blocks);
  //Mrows2.resize(_num_blocks);
  //Mcols2.resize(_num_blocks);
  std::vector<W> tmp1(_num_blocks);
  std::vector<W> tmp2(_num_blocks);
  std::vector<W> tmp3(_num_blocks);
  std::vector<W> tmp4(_num_blocks);
  
  for (const auto& [v, w] : _Mrow[r]) {
    if (w != 0) {
      tmp1[v] += w;
    }
  }
  for (const auto& [v, w] : _Mrow[s]) {
    if (w != 0) { 
      tmp2[v] += w;
    }
  }
  for (const auto& [v, w] : _Mcol[r]) {
    if (w != 0) {
      tmp3[v] += w;
    }
  }
  for (const auto& [v, w] : _Mcol[s]) {
    if (w != 0) {
      tmp4[v] += w;
    }
  }
  for (size_t v = 0; v < _num_blocks; v++) {
    if (tmp1[v] != 0) { 
      delta_entropy += tmp1[v] * std::log(static_cast<float> (tmp1[v]) / (_block_degrees_in[v] * _block_degrees_out[r]));
    }
    if (tmp2[v] != 0) {
      delta_entropy += tmp2[v] * std::log(static_cast<float> (tmp2[v]) / (_block_degrees_in[v] * _block_degrees_out[s]));
    }
    if (v != r && v != s) {
      if (tmp3[v] != 0) {
        delta_entropy += tmp3[v] * std::log(static_cast<float> (tmp3[v]) / (_block_degrees_out[v] * _block_degrees_in[r]));
      }
      if (tmp4[v] != 0) {
        delta_entropy += tmp4[v] * std::log(static_cast<float> (tmp4[v]) / (_block_degrees_out[v] * _block_degrees_in[s])); 
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
  std::vector<size_t>& block_map,
  std::vector<size_t>& remaining_blocks) 
{
  block_map.clear();
  block_map.resize(_num_blocks);
  std::iota(block_map.begin(), block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;

  while (num_merge < B_to_merge) {
    int mergeFrom = bestMerges[counter];
    int mergeTo = block_map[best_merge_for_each_block[bestMerges[counter]]];
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
  
  remaining_blocks = _unique(_partitions);
  block_map.clear();
  block_map.resize(_num_blocks, -1);
  for (size_t i = 0; i < remaining_blocks.size(); i++) {
    block_map[remaining_blocks[i]] = i;
  }

  for (auto& it : _partitions) {
    it = block_map[it];
  }

  return _num_blocks - B_to_merge;
} // end of carry_out_best_merges

template <typename W>
float Graph_P<W>::_compute_overall_entropy(
  std::vector<W>& M_r_row)
{

  M_r_row.clear();
  M_r_row.resize(_num_blocks);
  
  float data_S2 = 0;
  for (size_t i = 0; i < _num_blocks; i++) {
    std::fill(M_r_row.begin(), M_r_row.end(), 0);
    for (const auto& [v, w] : _Mrow[i]) {
      M_r_row[v] += w;
    }
    for (size_t v = 0; v < _num_blocks; v++) {
      if (M_r_row[v] != 0) {
        data_S2 -= M_r_row[v] * std::log(M_r_row[v] / (float)(_block_degrees_out[i] * _block_degrees_in[v]));
      }
    }
  }
 

  float model_S_term = (float)_num_blocks*_num_blocks/_E;
  float model_S = (float)(_E * (1 + model_S_term) * log(1 + model_S_term)) - 
                          (model_S_term * log(model_S_term)) + (_N * log(_num_blocks));

  return model_S + data_S2;
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
  const std::vector<W>& d_new
  )
{
  ////////////////////////////////////////////
  //check before update
  _partitions[ni] = s;
  // TODO: why can I just clear the row and col and only update the non zero
  // term?
  //

  // from the differet perspective
  _Mrow[r].clear();
  _Mrow[s].clear();
  _Mcol[r].clear();
  _Mcol[s].clear();
  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_row[i] != 0) {
      _Mrow[r].emplace_back(std::make_pair(i, M_r_row[i]));
    }
    if (M_s_row[i] != 0) {
      _Mrow[s].emplace_back(std::make_pair(i, M_s_row[i]));
    }
    if (i != r && i != s) { 
       _Mrow[i].erase(
        std::remove_if(_Mrow[i].begin(), _Mrow[i].end(), [r, s](const std::pair<size_t, W>& p) { 
            return (p.first == r) || (p.first == s);
        }),  
        _Mrow[i].end()
      ); 
  
      if (M_r_col[i] != 0) {
        _Mrow[i].emplace_back(std::make_pair(r, M_r_col[i]));
      }
      if (M_s_col[i] != 0) {
        _Mrow[i].emplace_back(std::make_pair(s, M_s_col[i]));
      }  
    }
  } 

  for (size_t i = 0; i < _num_blocks; i++) {
    if (M_r_col[i] != 0) {
      _Mcol[r].emplace_back(std::make_pair(i, M_r_col[i]));
    }
    if (M_s_col[i] != 0) {
      _Mcol[s].emplace_back(std::make_pair(i, M_s_col[i]));
    }
    if (i != r && i != s) {
      _Mcol[i].erase(
        std::remove_if(_Mcol[i].begin(), _Mcol[i].end(), [r, s](const std::pair<size_t, W>& p) {
            return (p.first == r) || (p.first == s);
        }),
        _Mcol[i].end()
      );
      if (M_r_row[i] != 0) {
        _Mcol[i].emplace_back(std::make_pair(r, M_r_row[i]));
      }
      if (M_s_row[i] != 0) {
        _Mcol[i].emplace_back(std::make_pair(s, M_s_row[i]));
      }     
    }
  
  }

  _block_degrees_out = d_out_new;
  _block_degrees_in = d_in_new;
  _block_degrees = d_new;

} // end of update_partition

template <typename W>
bool Graph_P<W>::_prepare_for_partition_next(
  float S,
  size_t& B_to_merge,
  float B_rate) 
{
  
  bool optimal_B_found = false;
  int index;
  
  if (S <= _old.overall_entropy_med) { 
    if (_old.num_blocks_med > _num_blocks) { 
      _old.partitions_large =       _old.partitions_med;
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
      _old.Mrow_small =             _old.Mrow_med;
      _old.Mcol_small =             _old.Mcol_med;
      _old.block_degree_small =     _old.block_degree_med;
      _old.block_degree_in_small =  _old.block_degree_in_med;
      _old.block_degree_out_small = _old.block_degree_out_med;
      _old.overall_entropy_small =  _old.overall_entropy_med;
      _old.num_blocks_small =       _old.num_blocks_med;  
    }
    _old.partitions_med =       _partitions; 
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
    B_to_merge =          (int)_num_blocks*B_rate;
    if (B_to_merge == 0)  optimal_B_found = true;
    _partitions =         _old.partitions_med;
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
        B_to_merge =          _old.num_blocks_large - next_B_to_try;
        _num_blocks =         _old.num_blocks_large;
        _partitions =         _old.partitions_large;
        _Mrow =               _old.Mrow_large;
        _Mcol =               _old.Mcol_large;
        _block_degrees =      _old.block_degree_large;
        _block_degrees_out =  _old.block_degree_out_large;
        _block_degrees_in =   _old.block_degree_in_large;
      }
      else {
        int next_B_to_try =   _old.num_blocks_small 
          + static_cast<int>(std::round((_old.num_blocks_med - _old.num_blocks_small) * 0.618));
        B_to_merge =          _old.num_blocks_med - next_B_to_try;
        _num_blocks =         _old.num_blocks_med;
        _partitions =         _old.partitions_med;
        _Mrow =               _old.Mrow_med;
        _Mcol =               _old.Mcol_med;
        _block_degrees =      _old.block_degree_med;
        _block_degrees_out =  _old.block_degree_out_med;
        _block_degrees_in =   _old.block_degree_in_med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


