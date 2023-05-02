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

namespace sgp {

template <typename W>
class Graph {

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
    Graph(const std::string& FileName) {
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
    std::vector<std::vector<std::pair<size_t, W>>> _inNeighbors;

    size_t _num_blocks;

    std::random_device _rd;
    std::default_random_engine _generator;
    
    struct OldPartition {
      std::vector<size_t> large;
      std::vector<size_t> med;
      std::vector<size_t> small;
    };
    
    struct OldInterblockEdgeCount {
      std::vector<W> large;
      std::vector<W> med;
      std::vector<W> small;
    };

    struct OldBlockDegrees {
      std::vector<W> large;
      std::vector<W> med;
      std::vector<W> small;
    };
    
    struct OldBlockDegreesOut {
      std::vector<W> large;
      std::vector<W> med;
      std::vector<W> small;
    };
    
    struct OldBlockDegreesIn {
      std::vector<W> large;
      std::vector<W> med;
      std::vector<W> small;
    };

    struct OldOverallEntropy {
      float large;
      float med;
      float small;
    };

    struct OldNumBlocks {
      size_t large;
      size_t med;
      size_t small;
    };

    // functions used internally
    void _initialize_edge_counts(
      const std::vector<size_t>& partitions,
      std::vector<W>& M, 
      std::vector<W>& d_out, 
      std::vector<W>& d_in, 
      std::vector<W>& d
    );
  
    void _propose_new_partition(
      size_t r,
      const std::vector< std::pair<size_t, W> >& neighbors_out,
      const std::vector< std::pair<size_t, W> >& neighbors_in,
      const std::vector<size_t>& b,
      const std::vector<W>& M,
      const std::vector<W>& d,
      bool agg_move,
      const std::default_random_engine& generator,
      size_t& s,
      W& k_out,
      W& k_in,
      W& k,
      std::vector< std::pair<size_t, W> >& neighbors,
      std::vector<float>& probabilities
    );

    void _compute_new_rows_cols_interblock_edge_count(
      const std::vector<W>& M,
      size_t r,
      size_t s,
      const std::vector<std::pair<size_t, W>>& b_out,
      const std::vector<std::pair<size_t, W>>& b_in,
      W count_self,
      bool agg_move,
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
      const std::vector<std::pair<size_t, W>>& b_out,
      const std::vector<std::pair<size_t, W>>& b_in,
      size_t s,
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
      OldPartition& old_b,
      OldInterblockEdgeCount& old_M,
      OldBlockDegrees& old_d,
      OldBlockDegreesOut& old_d_out,
      OldBlockDegreesIn& old_d_in,
      OldOverallEntropy& old_S,
      OldNumBlocks& old_B,
      float B_rate
    );

    // utility functions
    std::vector<size_t> _argsort(const std::vector<float>& arr) {
      std::vector<size_t> sorted_indices(arr.size());
      std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1,
      std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&arr](int i, int j){ return arr[i] < arr[j]; });  // Sort by arr values
      return sorted_indices;
    }

    std::vector<size_t> _unique(const std::vector<size_t>& arr) {
      // TODO: consider moving unique_arr to Graph private members
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


}; // end of class Graph


// function definitions
//
//
template <typename W>
void Graph<W>::load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv"); // open the file in read mode
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
  _inNeighbors.resize(_N);
  
  for (auto& e : _edges) {
    _out_neighbors[e.from-1].emplace_back(e.to-1, e.weight);
    _inNeighbors[e.to-1].emplace_back(e.from-1, e.weight);
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
std::vector<size_t> Graph<W>::partition() {
  
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

  OldPartition _old_b;
  OldInterblockEdgeCount _old_M;
  OldBlockDegrees _old_d;
  OldBlockDegreesOut _old_d_out;
  OldBlockDegreesIn _old_d_in;
  OldOverallEntropy _old_S;
  OldNumBlocks _old_B;

  _old_S.large = std::numeric_limits<float>::infinity();
  _old_S.med = std::numeric_limits<float>::infinity();
  _old_S.small = std::numeric_limits<float>::infinity();

  _old_B.large = 0;
  _old_B.med = 0;
  _old_B.small = 0;

  bool optimal_num_blocks_found = false;

  size_t num_blocks_to_merge = (size_t)_num_blocks * num_block_reduction_rate;

  std::vector<int> best_merge_for_each_block;
  std::vector<float> delta_entropy_for_each_block;
  std::vector<size_t> bestMerges;
  std::vector<size_t> block_map;
  std::vector<size_t> block_partition;

  std::vector< std::pair<size_t, W> > out_blocks; // {index, weight}
  std::vector< std::pair<size_t, W> > in_blocks;

  // proposal
  std::vector< std::pair<size_t, W> > neighbors;
  std::vector<float> probabilities;

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
  //auto block_merge_propose_time = 0;
  //auto block_merge_compute_new_time = 0;
  //auto block_merge_compute_entropy_time = 0;
  //auto nodal_update_time = 0;
  //auto nodal_update_propose_time = 0;
  //auto nodal_update_compute_new_time = 0;
  //auto nodal_update_compute_entropy_time = 0;
  
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
    auto block_merge_start = std::chrono::steady_clock::now();
    for (size_t current_block = 0; current_block < _num_blocks; current_block++) {
      for (size_t proposal_idx = 0; proposal_idx < num_agg_proposals_per_block; proposal_idx++) {
        
        out_blocks.clear(); 
        in_blocks.clear();
        for (size_t i = 0; i < _num_blocks; i++) {
          if (interblock_edge_count[_num_blocks*current_block + i] != 0) {
            out_blocks.emplace_back(i, interblock_edge_count[_num_blocks*current_block + i]);
          }   
          if (interblock_edge_count[_num_blocks*i + current_block] != 0) {
            in_blocks.emplace_back(i, interblock_edge_count[_num_blocks*i + current_block]);
          }  
        } 

        size_t proposal;
        W num_out_neighbor_edges;
        W num_in_neighbor_edges;
        W num_neighbor_edges;

        //auto block_merge_propose_start = std::chrono::steady_clock::now();
        _propose_new_partition(
          current_block,
          out_blocks,
          in_blocks,
          block_partition,
          interblock_edge_count,
          block_degrees,
          1,
          _generator,
          proposal,
          num_out_neighbor_edges,
          num_in_neighbor_edges,
          num_neighbor_edges,
          neighbors,
          probabilities
        );
        //auto block_merge_propose_end = std::chrono::steady_clock::now();
        //block_merge_propose_time += std::chrono::duration_cast<std::chrono::microseconds>
        //                              (block_merge_propose_end - block_merge_propose_start).count();

        interblock_edge_count_r_row_new.clear();
        interblock_edge_count_s_row_new.clear();
        interblock_edge_count_r_col_new.clear();
        interblock_edge_count_s_col_new.clear();

        W self_edge_count = interblock_edge_count[current_block*_num_blocks+current_block];

        //auto block_merge_compute_new_start = std::chrono::steady_clock::now();
        _compute_new_rows_cols_interblock_edge_count(
          interblock_edge_count,
          current_block,
          proposal,
          out_blocks,//
          in_blocks,//
          self_edge_count,
          1,
          interblock_edge_count_r_row_new,
          interblock_edge_count_s_row_new,
          interblock_edge_count_r_col_new,
          interblock_edge_count_s_col_new
        );
        //auto block_merge_compute_new_end = std::chrono::steady_clock::now();
        //block_merge_compute_new_time += std::chrono::duration_cast<std::chrono::microseconds>
        //                                  (block_merge_compute_new_end - block_merge_compute_new_start).count();

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
        
        //auto block_merge_compute_entropy_start = std::chrono::steady_clock::now();
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
        //auto block_merge_compute_entropy_end = std::chrono::steady_clock::now();
        //block_merge_compute_entropy_time += std::chrono::duration_cast<std::chrono::microseconds>
        //                                      (block_merge_compute_entropy_end - block_merge_compute_entropy_start).count();

        if (delta_entropy < delta_entropy_for_each_block[current_block]) {
          best_merge_for_each_block[current_block] = proposal;
          delta_entropy_for_each_block[current_block] = delta_entropy;
        }
      } // end proposal_idx
    } // end current_block
    //auto block_merge_end = std::chrono::steady_clock::now();
    //block_merge_time += std::chrono::duration_cast<std::chrono::milliseconds>(block_merge_end - block_merge_start).count();

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
      
        size_t proposal;
        W num_out_neighbor_edges;
        W num_in_neighbor_edges;
        W num_neighbor_edges;

        //auto nodal_update_propose_start = std::chrono::steady_clock::now();
        _propose_new_partition(
          current_block,
          _out_neighbors[current_node],
          _inNeighbors[current_node],
          partitions,
          interblock_edge_count,
          block_degrees,
          0,
          _generator,
          proposal,
          num_out_neighbor_edges,
          num_in_neighbor_edges,
          num_neighbor_edges,
          neighbors,
          probabilities
        );
        //auto nodal_update_propose_end = std::chrono::steady_clock::now();
        //nodal_update_propose_time += std::chrono::duration_cast<std::chrono::microseconds>
        //                              (nodal_update_propose_end - nodal_update_propose_start).count();
        
        if (proposal != current_block) {
          out_blocks.clear(); 
          in_blocks.clear();
          for (const auto& ele : _out_neighbors[current_node]) {
            out_blocks.emplace_back(partitions[ele.first], ele.second);
          }
          for (const auto& ele : _inNeighbors[current_node]) {
            in_blocks.emplace_back(partitions[ele.first], ele.second);
          }




          W self_edge_weight = 0;
          for (const auto& ele : _out_neighbors[current_node]) {
            if (ele.first == current_node) self_edge_weight += ele.second;
          }

          interblock_edge_count_r_row_new.clear();
          interblock_edge_count_s_row_new.clear();
          interblock_edge_count_r_col_new.clear();
          interblock_edge_count_s_col_new.clear();

          //auto nodal_update_compute_new_start = std::chrono::steady_clock::now();
          _compute_new_rows_cols_interblock_edge_count(
            interblock_edge_count,
            current_block,
            proposal,
            out_blocks, //
            in_blocks, //
            self_edge_weight,
            0,
            interblock_edge_count_r_row_new,
            interblock_edge_count_s_row_new,
            interblock_edge_count_r_col_new,
            interblock_edge_count_s_col_new
          );
          //auto nodal_update_compute_new_end = std::chrono::steady_clock::now();
          //nodal_update_compute_new_time += std::chrono::duration_cast<std::chrono::microseconds>
          //                                  (nodal_update_compute_new_end - nodal_update_compute_new_start).count();

          
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
                                    out_blocks,//
                                    in_blocks,//
                                    proposal,
                                    interblock_edge_count,
                                    interblock_edge_count_r_row_new,
                                    interblock_edge_count_r_col_new,
                                    block_degrees,
                                    block_degrees_new
                                  );
          } // calculate Hastings_correction

          //auto nodal_update_compute_entropy_start = std::chrono::steady_clock::now();          
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
          //auto nodal_update_compute_entropy_end = std::chrono::steady_clock::now();
          //nodal_update_compute_entropy_time += std::chrono::duration_cast<std::chrono::microseconds>
          //                                      (nodal_update_compute_entropy_end - nodal_update_compute_entropy_start).count();

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
        isfinite = isfinite && std::isfinite(_old_S.large);
        isfinite = isfinite && std::isfinite(_old_S.med);
        isfinite = isfinite && std::isfinite(_old_S.small);
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
    } // end itr (nodal update)
    //auto nodal_update_end = std::chrono::steady_clock::now();
    //nodal_update_time += std::chrono::duration_cast<std::chrono::milliseconds>(nodal_update_end - nodal_update_start).count();


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
                                 _old_b,
                                 _old_M,
                                 _old_d,
                                 _old_d_out,
                                 _old_d_in,
                                 _old_S,
                                 _old_B,
                                 num_block_reduction_rate
                               );

    if (verbose) {
      printf("Overall entropy: [%f, %f, %f] \n", _old_S.large, _old_S.med, _old_S.small);
      printf("Number of blocks: [%ld, %ld, %ld] \n", _old_B.large, _old_B.med, _old_B.small);
    }

  } // end while
  
  //std::cout << "block_merge_time: " << block_merge_time << std::endl;
  //std::cout << "block_merge_propose_time: " << block_merge_propose_time << std::endl;
  //std::cout << "block_merge_compute_new_time: " << block_merge_compute_new_time << std::endl;
  //std::cout << "block_merge_compute_entropy_time: " << block_merge_compute_entropy_time << std::endl;
  //std::cout << "nodal_update_time: " << nodal_update_time << std::endl;
  //std::cout << "nodal_update_propose_time: " << nodal_update_propose_time << std::endl;
  //std::cout << "nodal_update_compute_new_time: " << nodal_update_compute_new_time << std::endl;
  //std::cout << "nodal_update_compute_entropy_time: " << nodal_update_compute_entropy_time << std::endl;

  return partitions;
}

template <typename W>
void Graph<W>::_initialize_edge_counts(
  const std::vector<size_t>& partitions,
  std::vector<W>& M, 
  std::vector<W>& d_out, 
  std::vector<W>& d_in, 
  std::vector<W>& d
) {

  M.clear();
  M.resize(_num_blocks * _num_blocks, 0);

  // compute the initial interblock edge count
  for (size_t node = 0; node < _out_neighbors.size(); node++) {
    if (_out_neighbors[node].size() > 0) {
      size_t k1 = partitions[node]; // get the block of the current node
      for (auto& e: _out_neighbors[node]) {
        // get the block of the neighbor node
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
    for (size_t j = 0; j < _num_blocks; j++) {
      d_out[i] += M[i*_num_blocks + j]; // row
      d_in[i] += M[j*_num_blocks + i]; // col
    }
    d[i] = d_out[i] + d_in[i];
  }
} // end of initialize_edge_counts

template <typename W>
void Graph<W>::_propose_new_partition(
  size_t r,
  const std::vector< std::pair<size_t, W> >& neighbors_out,
  const std::vector< std::pair<size_t, W> >& neighbors_in,
  const std::vector<size_t>& b,
  const std::vector<W>& M,
  const std::vector<W>& d,
  bool agg_move,
  const std::default_random_engine& generator,
  size_t& s,
  W& k_out,
  W& k_in,
  W& k,
  std::vector< std::pair<size_t, W> >& neighbors,
  std::vector<float>& probabilities
) {

  size_t B = _num_blocks;

  neighbors = neighbors_out;
  neighbors.insert(neighbors.end(), neighbors_in.begin(), neighbors_in.end());

  k_out = 0;
  for (const auto& it : neighbors_out) k_out += it.second;
  k_in = 0;
  for (const auto& it : neighbors_in) k_in += it.second;
  k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, B-1); // 0 ~ B-1
  if (k == 0) { // this node has no neighbor, simply propose a block randomly
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }

  // create the probabilities array based on the edge weight of each neighbor
  probabilities.clear();
  for (const auto& n : neighbors) {
    probabilities.push_back( (float)n.second/k );
  }
  // create a discrete distribution based on the probabilities array
  std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
  int rand_neighbor = neighbors[distribution(const_cast<std::default_random_engine&>(generator))].first;
  int u = b[rand_neighbor];

  // propose a new block randomly
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float rand_num = uni_dist(const_cast<std::default_random_engine&>(generator));
  if ( rand_num <= (float)B/(d[u]+B) ) { // chance inversely prop. to block_degree
    if (agg_move) { // force proposal to be different from current block
      std::vector<int> candidates;
      for (int i = 0; i < B; i++) {
        if (i != r) candidates.push_back(i);
      }
      std::uniform_int_distribution<int> choice(0, candidates.size()-1);
      s = candidates[choice(const_cast<std::default_random_engine&>(generator))];
    }
    else{
      s = randint(const_cast<std::default_random_engine&>(generator));
    }
  }
  else { // propose by random draw from neighbors of block partition[rand_neighbor]
    std::vector<float> multinomial_prob(B);
    float multinomial_prob_sum = 0;
    for (size_t i = 0; i < B; i++) {
      multinomial_prob[i] = (float)(M[u*B + i] + M[i*B + u])/d[u];
      multinomial_prob_sum += multinomial_prob[i];
    }
    if (agg_move) { // force proposal to be different from current block
      multinomial_prob[r] = 0;
      // recalculate
      multinomial_prob_sum = 0;
      for (size_t i = 0; i < B; i++) {
        multinomial_prob_sum += multinomial_prob[i];
      }
      // check
      if (multinomial_prob_sum == 0) { // the current block has no neighbors. randomly propose a different block
        std::vector<int> candidates;
        for (size_t i = 0; i < B; i++) {
          if (i != r) candidates.push_back(i);
        }
        std::uniform_int_distribution<int> choice(0, candidates.size()-1);
        s = candidates[choice(const_cast<std::default_random_engine&>(generator))];
        return;
      }
      else {
        for (auto& it : multinomial_prob) it /= multinomial_prob_sum;
      }
    }
    std::discrete_distribution<int> multinomial(multinomial_prob.begin(), multinomial_prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }
}

template <typename W>
void Graph<W>::_compute_new_rows_cols_interblock_edge_count(
  const std::vector<W>& M,
  size_t r,
  size_t s,
  const std::vector< std::pair<size_t, W> >& b_out,
  const std::vector< std::pair<size_t, W> >& b_in,
  W count_self,
  bool agg_move,
  std::vector<W>& M_r_row,
  std::vector<W>& M_s_row,
  std::vector<W>& M_r_col,
  std::vector<W>& M_s_col
) {

  size_t B = _num_blocks;
  
  M_r_row.clear();
  M_r_col.clear();
  M_s_row.clear();
  M_s_col.clear();

  if (agg_move) { // the r row and column are simply empty after this merge move
    M_r_row.resize(B, 0);
    M_r_col.resize(B, 0);
  }
  else {
    M_r_row.resize(B, 0);
    M_r_col.resize(B, 0);
    // TODO: taskflow::for_each_index
    for (size_t i = 0; i < B; i++) {
      M_r_row[i] = M[r*B + i];
      M_r_col[i] = M[i*B + r];
    }
    
    W count_in_sum_r = 0;
    W count_out_sum_r = 0; 
    
    for (size_t i = 0; i < b_in.size(); i++) {
      if (b_in[i].first == r) {
        count_in_sum_r += b_in[i].second;
      }
      M_r_col[ b_in[i].first ] -= b_in[i].second;
    }
    
    for (size_t i = 0; i < b_out.size(); i++) {
      if (b_out[i].first == r) {
        count_out_sum_r += b_out[i].second;
      }
      M_r_row[ b_out[i].first ] -= b_out[i].second;
    }
    
    M_r_row[r] -= count_in_sum_r;
    M_r_row[s] += count_in_sum_r;
    M_r_col[r] -= count_out_sum_r;
    M_r_col[s] += count_out_sum_r;
  }
  
  M_s_row.resize(B, 0);
  M_s_col.resize(B, 0);
  for (size_t i = 0; i < B; i++) {
    M_s_row[i] = M[s*B + i];
    M_s_col[i] = M[i*B + s];
  }

  W count_in_sum_s = 0;
  W count_out_sum_s = 0; 

  for (size_t i = 0; i < b_in.size(); i++) {
    if (b_in[i].first == s) {
      count_in_sum_s += b_in[i].second;
    }
    M_s_col[ b_in[i].first ] += b_in[i].second;
  }

  for (size_t i = 0; i < b_out.size(); i++) {
    if ( b_out[i].first == s) {
      count_out_sum_s += b_out[i].second;
    }
    M_s_row[ b_out[i].first ] += b_out[i].second;
  }
  
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
void Graph<W>::_compute_new_block_degree(
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
float Graph<W>::_compute_delta_entropy(
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
size_t Graph<W>::_carry_out_best_merges(
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
float Graph<W>::_compute_overall_entropy(
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
float Graph<W>::_compute_Hastings_correction(
  const std::vector< std::pair<size_t, W> >& b_out,
  const std::vector< std::pair<size_t, W> >& b_in,
  size_t s,
  const std::vector<W>& M,
  const std::vector<W>& M_r_row,
  const std::vector<W>& M_r_col,
  const std::vector<W>& d,
  const std::vector<W>& d_new
) {
  size_t B = _num_blocks;
 
  float p_forward = 0;
  float p_backward = 0;
  for (size_t i = 0; i < b_out.size(); i++) {
    p_forward += (float)b_out[i].second * (M[b_out[i].first*B + s] + M[s*B + b_out[i].first] + 1) / 
                    (d[b_out[i].first] + B);
    p_backward += (float)b_out[i].second * (M_r_row[b_out[i].first] + M_r_col[b_out[i].first] + 1) / 
                    (d_new[b_out[i].first] + B);
  }
  for (size_t i = 0; i < b_in.size(); i++) {
    p_forward += (float)b_in[i].second * (M[b_in[i].first*B + s] + M[s*B + b_in[i].first] + 1) / 
                      (d[b_in[i].first] + B);
    p_backward += (float)b_in[i].second * (M_r_row[b_in[i].first] + M_r_col[b_in[i].first] + 1) / 
                      (d_new[b_in[i].first] + B);
  }
  return p_backward / p_forward;
} // end of compute_Hastings_correction


template <typename W>
void Graph<W>::_update_partition(
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
bool Graph<W>::_prepare_for_partition_next(
  float S,
  std::vector<size_t>& b,
  std::vector<W>& M,
  std::vector<W>& d,
  std::vector<W>& d_out,
  std::vector<W>& d_in,
  size_t& B,
  size_t& B_to_merge,
  OldPartition& old_b,
  OldInterblockEdgeCount& old_M,
  OldBlockDegrees& old_d,
  OldBlockDegreesOut& old_d_out,
  OldBlockDegreesIn& old_d_in,
  OldOverallEntropy& old_S,
  OldNumBlocks& old_B,
  float B_rate
) {
  bool optimal_B_found = false;
  int index;
  
  if (S <= old_S.med) {  // if the current overall entropy is the best so far
    if ( old_B.med > B) { 
      old_b.large = old_b.med;
      old_M.large = old_M.med;
      old_d.large = old_d.med;
      old_d_in.large = old_d_in.med;
      old_d_out.large = old_d_out.med;
      old_S.large = old_S.med;
      old_B.large = old_B.med;
    }
    else {
      old_b.small = old_b.med;
      old_M.small = old_M.med;
      old_d.small = old_d.med;
      old_d_in.small = old_d_in.med;
      old_d_out.small = old_d_out.med;
      old_S.small = old_S.med;
      old_B.small = old_B.med;  
    }
    old_b.med = b; 
    old_M.med = M;
    old_d.med = d;
    old_d_in.med = d_in;
    old_d_out.med = d_out;
    old_S.med = S;
    old_B.med = B;
  }
  else {
    if ( old_B.med > B) {
      old_b.small = b;
      old_M.small = M;
      old_d.small = d;
      old_d_in.small = d_in;
      old_d_out.small = d_out;
      old_S.small = S;
      old_B.small = B; 
    }
    else {
      old_b.large = b;
      old_M.large = M;
      old_d.large = d;
      old_d_in.large = d_in;
      old_d_out.large = d_out;
      old_S.large = S;
      old_B.large = B;
    }
  }
 
  if (std::isinf(old_S.small)) {
    B_to_merge = (int)B*B_rate;
    if (B_to_merge == 0) optimal_B_found = true;
    b = old_b.med;
    M = old_M.med;
    d = old_d.med;
    d_out = old_d_out.med;
    d_in = old_d_in.med; 
  }
  else {
    // golden ratio search bracket established
    // we have found the partition with the optimal number of blocks
    if (old_B.large - old_B.small == 2) {
      optimal_B_found = true;
      B = old_B.med;
      b = old_b.med;
    }
    // not done yet, find the next number of block to try according to the golden ratio search
    // the higher segment in the bracket is bigger
    else {
      // the higher segment in the bracket is bigger
      if ((old_B.large-old_B.med) >= (old_B.med-old_B.small)) {  
        int next_B_to_try = old_B.med + static_cast<int>(round((old_B.large - old_B.med) * 0.618));
        B_to_merge = old_B.large - next_B_to_try;
        B = old_B.large;
        b = old_b.large;
        M = old_M.large;
        d = old_d.large;
        d_out = old_d_out.large;
        d_in = old_d_in.large;
      }
      else {
        int next_B_to_try = old_B.small + static_cast<int>(round((old_B.med - old_B.small) * 0.618));
        B_to_merge = old_B.med - next_B_to_try;
        B = old_B.med;
        b = old_b.med;
        M = old_M.med;
        d = old_d.med;
        d_out = old_d_out.med;
        d_in = old_d_in.med;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp


