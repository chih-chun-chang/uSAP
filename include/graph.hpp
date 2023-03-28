#pragma once 

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <map>
#include <set>
#include <memory>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

#include "utils.hpp"

namespace sgp {

template <typename W>
class Graph {

  public: 

    struct Edge {
      int from;
      int to;
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

    // function used by users
    void load_graph_from_tsv(const std::string& FileName);
    std::vector<size_t> partition();    

    // constructor
    Graph(const std::string& FileName) {
      load_graph_from_tsv(FileName);
    }

    // results
    std::vector<size_t> truePartitions;            // partition ground truth

  private:

    size_t _N;                                      // number of node
    size_t _E;                                      // number of edge
    
    std::vector<Edge> _edges;
    std::vector<std::vector<Edge>> _adjLists;       // N*N
    std::vector<std::vector<std::pair<size_t, W>>> _outNeighbors;   // N*(each node's #outNeighbors )
    std::vector<std::vector<std::pair<size_t, W>>> _inNeighbors;    // N*(each node's #inNeighbors ) 
    //std::vector<size_t> _truePartitions;            // partition ground truth

    size_t _num_blocks;                             // number of blocks

    // TODO: optimized this
    std::unordered_map< int, std::vector<size_t> > old_partition;
    std::unordered_map< int, std::vector<W> > old_interblock_edge_count;
    std::unordered_map< int, std::vector<W> > old_block_degrees;
    std::unordered_map< int, std::vector<W> > old_block_degrees_out;
    std::unordered_map< int, std::vector<W> > old_block_degrees_in;
    std::unordered_map< int, double > old_overall_entropy;
    std::unordered_map< int, size_t >old_num_blocks;

    // functions used internally
    void _initialize_edge_counts(const std::vector<size_t>& partitions,
                                 std::vector<W>& M, 
                                 std::vector<W>& d_out, 
                                 std::vector<W>& d_in, 
                                 std::vector<W>& d);

    void _propose_new_partition(const size_t& r,
                                const std::vector< std::pair<size_t, W> >& neighbors_out,
                                const std::vector< std::pair<size_t, W> >& neighbors_in,
                                const std::vector<size_t>& b,
                                const std::vector<W>& M,
                                const std::vector<W>& d,
                                const bool& agg_move,
                                // TODO::
                                //const std::mt19937& generator,
                                // for return
                                size_t& s,
                                W& k_out,
                                W& k_in,
                                W& k);

    void _compute_new_rows_cols_interblock_edge_count_matrix(const std::vector<W>& M,
                                                             const size_t& r,
                                                             const size_t& s,
                                                             const std::vector<size_t>& b_out,
                                                             const std::vector<W>& count_out,
                                                             const std::vector<size_t>& b_in,
                                                             const std::vector<W>& count_in,
                                                             const W& count_self,
                                                             const bool& agg_move,
                                                             // for return
                                                             std::vector<W>& M_r_row,
                                                             std::vector<W>& M_s_row,
                                                             std::vector<W>& M_r_col,
                                                             std::vector<W>& M_s_col);
    
    void _compute_new_block_degree(const size_t& r,
                                   const size_t& s,
                                   const std::vector<W>& d_out,
                                   const std::vector<W>& d_in,
                                   const std::vector<W>& d,
                                   const W& k_out,
                                   const W& k_in,
                                   const W& k,
                                   // for return
                                   std::vector<W>& d_out_new,
                                   std::vector<W>& d_in_new,
                                   std::vector<W>& d_new);
                                  
    double _compute_delta_entropy(const size_t& r,
                                  const size_t& s,
                                  const std::vector<W>& M,
                                  // TODO
                                  std::vector<W> M_r_row,
                                  std::vector<W> M_s_row,
                                  std::vector<W> M_r_col,
                                  std::vector<W> M_s_col,
                                  const std::vector<W>& d_out,
                                  const std::vector<W>& d_in,
                                  const std::vector<W>& d_out_new,
                                  const std::vector<W>& d_in_new);        
     
    size_t _carry_out_best_merges(const std::vector<double>& delta_entropy_for_each_block,
                                  const std::vector<int>& best_merge_for_each_block,
                                  const size_t& B_to_merge,
                                  std::vector<size_t>& b);

    double _compute_overall_entropy(const std::vector<W>& M,
                                    const std::vector<W>& d_out,
                                    const std::vector<W>& d_in); 

    double _compute_Hastings_correction(const std::vector<size_t>& b_out,
                                        const std::vector<W>& count_out,
                                        const std::vector<size_t>& b_in,
                                        const std::vector<W>& count_in,
                                        const size_t& s,
                                        const std::vector<W>& M,
                                        const std::vector<W>& M_r_row,
                                        const std::vector<W>& M_r_col,
                                        const std::vector<W>& d,
                                        const std::vector<W>& d_new);

    void _update_partition(const size_t& ni,
                           const size_t& r,
                           const size_t& s,
                           const std::vector<W>& M_r_row,
                           const std::vector<W>& M_s_row,
                           const std::vector<W>& M_r_col,
                           const std::vector<W>& M_s_col,
                           const std::vector<W>& d_out_new,
                           const std::vector<W>& d_in_new,
                           const std::vector<W>& d_new,
                           // for return
                           std::vector<size_t>& b,
                           std::vector<W>& M,
                           std::vector<W>& d_out,
                           std::vector<W>& d_in,
                           std::vector<W>& d);

    bool _prepare_for_partition_on_next_num_blocks(double S,
                                                   std::vector<size_t>& b,
                                                   std::vector<W>& M,
                                                   std::vector<W>& d,
                                                   std::vector<W>& d_out,
                                                   std::vector<W>& d_in,
                                                   size_t & B,
                                                   size_t & B_to_merge,
                                                   std::unordered_map< int, std::vector<size_t> >& old_b,
                                                   std::unordered_map< int, std::vector<W> >& old_M,
                                                   std::unordered_map< int, std::vector<W> >& old_d,
                                                   std::unordered_map< int, std::vector<W> >& old_d_out,
                                                   std::unordered_map< int, std::vector<W> >& old_d_in,
                                                   std::unordered_map< int, double >& old_S,
                                                   std::unordered_map< int, size_t >& old_B,
                                                   double B_rate);

}; // end of class Graph


// function definitions
//
//
template <typename W>
void Graph<W>::load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv"); // open the file in read mode
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    exit(EXIT_FAILURE);
  }

  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;
  while (getline(file, line)) {
    Edge  edge;
    size_t start = 0;
    size_t tab_pos = line.find('\t');
    edge.from =  std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    edge.to = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    edge.weight = static_cast<W>(std::stof(line.substr(start, tab_pos - start)));
    _edges.push_back(edge);
  }
  file.close();

  auto maxElement = std::max_element(_edges.begin(), _edges.end(),
                        [](const Edge& e1, const Edge& e2) {
                          return e1.from < e2.from;
                        });
  _N = maxElement->from;
  _E = _edges.size();

  _adjLists.resize(_N, std::vector<Edge>(_N));
  _outNeighbors.resize(_N, std::vector<std::pair<size_t, W>>(0));
  _inNeighbors.resize(_N, std::vector<std::pair<size_t, W>>(0));
  
  for (auto& e : _edges) {
    _adjLists[e.from-1][e.to-1] = e;
    _outNeighbors[e.from-1].push_back({e.to-1, e.weight});
    _inNeighbors[e.to-1].push_back({e.from-1, e.weight});
  }

  // load the true partition
  std::ifstream true_file(FileName + "_truePartition.tsv");
  if (!true_file.is_open()) {
    std::cerr << "Unable to open file!\n";
    exit(EXIT_FAILURE);
  }
  // format: node i \t block
  while (getline(true_file, line)) {
    size_t start = 0;
    size_t tab_pos = line.find('\t');
    size_t i = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    size_t block = std::stoi(line.substr(start, tab_pos - start));
    truePartitions.push_back(block-1);
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

  _initialize_edge_counts(partitions, interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees);

  // TODO: optimized this
  old_overall_entropy[0] = std::numeric_limits<double>::infinity();
  old_overall_entropy[1] = std::numeric_limits<double>::infinity();
  old_overall_entropy[2] = std::numeric_limits<double>::infinity();


  bool optimal_num_blocks_found = false;

  size_t num_blocks_to_merge = (size_t)_num_blocks * num_block_reduction_rate;

  std::vector<int> best_merge_for_each_block;
  std::vector<double> delta_entropy_for_each_block;
  std::vector<size_t> block_partition;

  std::vector< std::pair<size_t, W> > out_blocks; // {index, weight}
  std::vector< std::pair<size_t, W> > in_blocks;
  std::vector<size_t> out_blocks_i;
  std::vector<W> out_blocks_w;
  std::vector<size_t> in_blocks_i;
  std::vector<W> in_blocks_w;

  //std::random_device rd; // seed the random number generator
  //std::mt19937 generator(rd());     

  std::vector<W> new_interblock_edge_count_current_block_row;
  std::vector<W> new_interblock_edge_count_new_block_row;
  std::vector<W> new_interblock_edge_count_current_block_col;
  std::vector<W> new_interblock_edge_count_new_block_col;

  std::vector<W> block_degrees_out_new;
  std::vector<W> block_degrees_in_new;
  std::vector<W> block_degrees_new;

  while (!optimal_num_blocks_found) {
    printf("\nMerging down blocks from %ld to %ld\n", _num_blocks, _num_blocks - num_blocks_to_merge);
    // init record for the round
    best_merge_for_each_block.clear();
    best_merge_for_each_block.resize(_num_blocks, -1);
    delta_entropy_for_each_block.clear();
    delta_entropy_for_each_block.resize(_num_blocks, std::numeric_limits<double>::infinity());
    block_partition.clear();
    block_partition.resize(_num_blocks, 0);
    std::iota(block_partition.begin(), block_partition.end(), 0);

    for (size_t current_block = 0; current_block < _num_blocks; current_block++) {
      for (size_t proposal_idx = 0; proposal_idx < num_agg_proposals_per_block; proposal_idx++) {
        
        out_blocks.clear(); 
        in_blocks.clear();
        for (size_t i = 0; i < _num_blocks; i++) {
          if (interblock_edge_count[_num_blocks*current_block + i] != 0) {
            out_blocks.push_back({i, interblock_edge_count[_num_blocks*current_block + i]});
          }   
          if (interblock_edge_count[_num_blocks*i + current_block] != 0) {
            in_blocks.push_back({i, interblock_edge_count[_num_blocks*i + current_block]});
          }  
        } 

        size_t proposal;
        W num_out_neighbor_edges;
        W num_in_neighbor_edges;
        W num_neighbor_edges;

        _propose_new_partition(current_block,
                               out_blocks,
                               in_blocks,
                               block_partition,
                               interblock_edge_count,
                               block_degrees,
                               1,
                               //generator,
                               proposal,
                               num_out_neighbor_edges,
                               num_in_neighbor_edges,
                               num_neighbor_edges);
  
        out_blocks_i.clear();
        out_blocks_w.clear();
        in_blocks_i.clear();
        in_blocks_w.clear();
        for (auto ele : out_blocks) {
          out_blocks_i.push_back(ele.first);
          out_blocks_w.push_back(ele.second);
        }
        for (auto ele : in_blocks) {
          in_blocks_i.push_back(ele.first);
          in_blocks_w.push_back(ele.second);
        }

        new_interblock_edge_count_current_block_row.clear();
        new_interblock_edge_count_new_block_row.clear();
        new_interblock_edge_count_current_block_col.clear();
        new_interblock_edge_count_new_block_col.clear();

        _compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count,
                                                            current_block,
                                                            proposal,
                                                            out_blocks_i,
                                                            out_blocks_w,
                                                            in_blocks_i,
                                                            in_blocks_w,
                                                            interblock_edge_count[current_block*_num_blocks+current_block],
                                                            1,
                                                            new_interblock_edge_count_current_block_row,
                                                            new_interblock_edge_count_new_block_row,
                                                            new_interblock_edge_count_current_block_col,
                                                            new_interblock_edge_count_new_block_col);

        block_degrees_out_new.clear();
        block_degrees_in_new.clear();
        block_degrees_new.clear();

        _compute_new_block_degree(current_block,
                                  proposal,
                                  block_degrees_out,
                                  block_degrees_in,
                                  block_degrees,
                                  num_out_neighbor_edges,
                                  num_in_neighbor_edges,
                                  num_neighbor_edges,
                                  block_degrees_out_new,
                                  block_degrees_in_new,
                                  block_degrees_new);

         double delta_entropy = _compute_delta_entropy(current_block,
                                                       proposal,
                                                       interblock_edge_count,
                                                       new_interblock_edge_count_current_block_row,
                                                       new_interblock_edge_count_new_block_row,
                                                       new_interblock_edge_count_current_block_col,
                                                       new_interblock_edge_count_new_block_col,
                                                       block_degrees_out,
                                                       block_degrees_in,
                                                       block_degrees_out_new,
                                                       block_degrees_in_new);
       
        if (delta_entropy < delta_entropy_for_each_block[current_block]) {
          best_merge_for_each_block[current_block] = proposal;
          delta_entropy_for_each_block[current_block] = delta_entropy;
        }
      } // end proposal_idx
    } // end current_block

    _num_blocks = _carry_out_best_merges(delta_entropy_for_each_block,
                                         best_merge_for_each_block,
                                         num_blocks_to_merge,
                                         partitions);
    
    //
    //
    //

    interblock_edge_count.clear();
    block_degrees_out.clear();
    block_degrees_in.clear();
    block_degrees.clear();
    _initialize_edge_counts(partitions, interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees);

    int total_num_nodal_moves = 0;
    //TODO: move outside
    std::vector<double> itr_delta_entropy(max_num_nodal_itr, 0.0);

    double overall_entropy = _compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in);
    printf("overall_entropy: %f\n", overall_entropy);
  
    for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {

      int num_nodal_moves = 0;
      itr_delta_entropy[itr] = 0;

      for (size_t current_node = 0; current_node < _N; current_node++) {
      
        size_t current_block = partitions[current_node];
      
        out_blocks.clear();
        in_blocks.clear();
        for (auto& e : _outNeighbors[current_node]) {
          out_blocks.push_back({e.first, e.second});
        }
        for (auto& e : _inNeighbors[current_node]) {
          in_blocks.push_back({e.first, e.second});
        }

        size_t proposal;
        W num_out_neighbor_edges;
        W num_in_neighbor_edges;
        W num_neighbor_edges;

        _propose_new_partition(current_block,
                               out_blocks,
                               in_blocks,
                               partitions,
                               interblock_edge_count,
                               block_degrees,
                               0,
                               // TODO
                               //generator,
                               proposal,
                               num_out_neighbor_edges,
                               num_in_neighbor_edges,
                               num_neighbor_edges);
        
        if (proposal != current_block) {
          // TODO: I thing this is redundant
          // NOTE: the key is partition block!!
          //
          std::map<size_t, W> blocks_out_temp;
          std::map<size_t, W> blocks_in_temp;
          for (auto& ele : _outNeighbors[current_node]) {
            //std::cout << "ele.first: " << ele.first << " partitions[ele.first]  "  << partitions[ele.first]<< " ele.second: " << ele.second << std::endl;
            blocks_out_temp[partitions[ele.first]] += ele.second;
          }   
          for (auto& ele : _inNeighbors[current_node]) {
            //std::cout << "ele.first: " << ele.first << " ele.second: " << ele.second << std::endl;
            blocks_in_temp[partitions[ele.first]] += ele.second;
          }   

          std::vector<size_t> blocks_out;
          std::vector<W> count_out;
          std::vector<size_t> blocks_in;
          std::vector<W> count_in; 
    
          for (const auto& [key, value] : blocks_out_temp) {
            blocks_out.push_back(key);
            count_out.push_back(value);
          }   
          for (const auto& [key, value] : blocks_in_temp) {
            blocks_in.push_back(key);
            count_in.push_back(value);
          }

          W self_edge_weight = 0;
          for (auto& ele : _outNeighbors[current_node]) {
            if (ele.first == current_node) self_edge_weight += ele.second;
          }

          new_interblock_edge_count_current_block_row.clear();
          new_interblock_edge_count_new_block_row.clear();
          new_interblock_edge_count_current_block_col.clear();
          new_interblock_edge_count_new_block_col.clear();

          _compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count,
                                                              current_block,
                                                              proposal,
                                                              blocks_out,
                                                              count_out,
                                                              blocks_in,
                                                              count_in,
                                                              self_edge_weight,
                                                              0,
                                                              new_interblock_edge_count_current_block_row,
                                                              new_interblock_edge_count_new_block_row,
                                                              new_interblock_edge_count_current_block_col,
                                                              new_interblock_edge_count_new_block_col);
          
          block_degrees_out_new.clear();
          block_degrees_in_new.clear();
          block_degrees_new.clear();

          _compute_new_block_degree(current_block,
                                    proposal,
                                    block_degrees_out,
                                    block_degrees_in,
                                    block_degrees,
                                    num_out_neighbor_edges,
                                    num_in_neighbor_edges,
                                    num_neighbor_edges,
                                    block_degrees_out_new,
                                    block_degrees_in_new,
                                    block_degrees_new);

          double Hastings_correction = 1.0;
          if (num_neighbor_edges > 0) {
            Hastings_correction = _compute_Hastings_correction(blocks_out,
                                                               count_out,
                                                               blocks_in,
                                                               count_in,
                                                               proposal,
                                                               interblock_edge_count,
                                                               new_interblock_edge_count_current_block_row,
                                                               new_interblock_edge_count_current_block_col,
                                                               block_degrees,
                                                               block_degrees_new);
          } // calculate Hastings_correction
          
          double delta_entropy = _compute_delta_entropy(current_block,
                                                        proposal,
                                                        interblock_edge_count,
                                                        new_interblock_edge_count_current_block_row,
                                                        new_interblock_edge_count_new_block_row,
                                                        new_interblock_edge_count_current_block_col,
                                                        new_interblock_edge_count_new_block_col,
                                                        block_degrees_out,
                                                        block_degrees_in,
                                                        block_degrees_out_new,
                                                        block_degrees_in_new);

          double p_accept = std::min(std::exp(-beta * delta_entropy)*Hastings_correction, 1.0);


          // TODO generator
          std::random_device rd; // seed the random number generator
          std::mt19937 gen(rd());
          std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
          double rand_num = uni_dist(gen);
          if ( rand_num <= p_accept) {
            total_num_nodal_moves++;
            num_nodal_moves++;
            itr_delta_entropy[itr] += delta_entropy;
            // bugs here
            _update_partition(current_node,
                              current_block,
                              proposal,
                              new_interblock_edge_count_current_block_row,
                              new_interblock_edge_count_new_block_row,
                              new_interblock_edge_count_current_block_col,
                              new_interblock_edge_count_new_block_col,
                              block_degrees_out_new,
                              block_degrees_in_new,
                              block_degrees_new,
                              partitions,
                              interblock_edge_count,
                              block_degrees_out,
                              block_degrees_in,
                              block_degrees);
          }
        } // end if 
      } // end current_node
      double oe = overall_entropy = _compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in);
      printf("Itr: %ld, number of nodal moves: %d, delta S: %.5f, overall_entropy:%f \n",itr, num_nodal_moves, itr_delta_entropy[itr] / double(overall_entropy), oe);
      if (itr >= (delta_entropy_moving_avg_window - 1)) {
        bool isfinite = true;
        for (const auto& [key, value] : old_overall_entropy) {
          isfinite = isfinite && std::isfinite(value);
        }   
        double mean = 0;
        for (int i = itr - delta_entropy_moving_avg_window + 1; i < itr; i++) {
          mean += itr_delta_entropy[i];
        }   
        mean /= (double)(delta_entropy_moving_avg_window - 1); 
        if (!isfinite) {
          if (-mean < (delta_entropy_threshold1 * overall_entropy)) {
            printf("golden ratio bracket is not yet established: %f %f\n", -mean, delta_entropy_threshold1 * overall_entropy);
            break;
          }   
        }   
        else {
          if (-mean < (delta_entropy_threshold2 * overall_entropy)) {
            printf("golden ratio bracket is established: %f \n", -mean);
            break;
          }   
        }   
      }   
    } // end itr

    overall_entropy = _compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in);

    printf("Total number of nodal moves: %d, overall_entropy: %.2f\n", total_num_nodal_moves, overall_entropy);


    optimal_num_blocks_found = _prepare_for_partition_on_next_num_blocks(overall_entropy,
                                                                        partitions,
                                                                        interblock_edge_count,
                                                                        block_degrees,
                                                                        block_degrees_out,
                                                                        block_degrees_in,
                                                                        _num_blocks,
                                                                        num_blocks_to_merge,
                                                                        old_partition,
                                                                        old_interblock_edge_count,
                                                                        old_block_degrees,
                                                                        old_block_degrees_out,
                                                                        old_block_degrees_in,
                                                                        old_overall_entropy,
                                                                        old_num_blocks,
                                                                        num_block_reduction_rate);

    printf("Overall entropy: [%f, %f, %f] \n", old_overall_entropy[0], old_overall_entropy[1], old_overall_entropy[2]);
    printf("Number of blocks: [%ld, %ld, %ld] \n", old_num_blocks[0], old_num_blocks[1], old_num_blocks[2]);


  } // end while





  return partitions;
}


template <typename W>
void Graph<W>::_initialize_edge_counts(const std::vector<size_t>& partitions,
                                       std::vector<W>& M, 
                                       std::vector<W>& d_out, 
                                       std::vector<W>& d_in, 
                                       std::vector<W>& d) {

  M.clear();
  M.resize(_num_blocks * _num_blocks, 0);

  // compute the initial interblock edge count
  for (size_t node = 0; node < _outNeighbors.size(); node++) {
    if (_outNeighbors[node].size() > 0) {
      size_t k1 = partitions[node]; // get the block of the current node
      // TODO: improve this
      std::map<size_t, W> out;
      for (auto& e: _outNeighbors[node]) {
        int key = partitions[e.first]; // get the block of the neighbor node
        out[key] += e.second;
      }
      for (const auto& [k2, count] : out) {
        M[k1*_num_blocks + k2] += count;
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
void Graph<W>::_propose_new_partition(const size_t& r,
                                      const std::vector< std::pair<size_t, W> >& neighbors_out,
                                      const std::vector< std::pair<size_t, W> >& neighbors_in,
                                      const std::vector<size_t>& b,
                                      const std::vector<W>& M,
                                      const std::vector<W>& d,
                                      const bool& agg_move,
                                      //const std::mt19937& generator,
                                      // for return
                                      size_t& s,
                                      W& k_out,
                                      W& k_in,
                                      W& k) {

  size_t B = _num_blocks;

  std::random_device rd; // seed the random number generator
  std::mt19937 generator(rd());

  // TODO: improve this
  std::vector< std::pair<size_t, W> > neighbors(neighbors_out);
  neighbors.insert(neighbors.end(), neighbors_in.begin(), neighbors_in.end());

  k_out = 0;
  for (auto& it : neighbors_out) k_out += it.second;
  k_in = 0;
  for (auto& it : neighbors_in) k_in += it.second;
  k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, B-1); // 0 ~ B-1
  if (k == 0) { // this node has no neighbor, simply propose a block randomly
    s = randint(generator);
    return;
  }

  // TODO: avoid creating vector
  // create the probabilities array based on the edge weight of each neighbor
  std::vector<double> probabilities;
  for (auto& n : neighbors) {
    probabilities.push_back( (double)n.second/k );
  }
  // create a discrete distribution based on the probabilities array
  std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
  int rand_neighbor = neighbors[distribution(generator)].first;
  int u = b[rand_neighbor];

  // propose a new block randomly
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
  double rand_num = uni_dist(generator);
  if ( rand_num <= (double)B/(d[u]+B) ) { // chance inversely prop. to block_degree
    if (agg_move) { // force proposal to be different from current block
      std::vector<int> candidates;
      for (int i = 0; i < B; i++) {
        if (i != r) candidates.push_back(i);
      }
      std::uniform_int_distribution<int> choice(0, candidates.size()-1);
      s = candidates[choice(generator)];
    }
    else{
      s = randint(generator);
    }
  }
  else { // propose by random draw from neighbors of block partition[rand_neighbor]
    std::vector<double> multinomial_prob(B);
    double multinomial_prob_sum = 0;
    for (int i = 0; i < B; i++) {
      multinomial_prob[i] = (double)(M[u*B + i] + M[i*B + u])/d[u];
      multinomial_prob_sum += multinomial_prob[i];
    }
    if (agg_move) { // force proposal to be different from current block
      multinomial_prob[r] = 0;
      // recalculate
      multinomial_prob_sum = 0;
      for (int i = 0; i < B; i++) {
        multinomial_prob_sum += multinomial_prob[i];
      }
      // check
      if (multinomial_prob_sum == 0) { // the current block has no neighbors. randomly propose a different block
        std::vector<int> candidates;
        for (int i = 0; i < B; i++) {
          if (i != r) candidates.push_back(i);
        }
        std::uniform_int_distribution<int> choice(0, candidates.size()-1);
        s = candidates[choice(generator)];
        return;
      }
      else {
        for (auto& it : multinomial_prob) it /= multinomial_prob_sum;
      }
    }
    std::vector<int> nonzero_index;
    std::vector<double> nonzero_prob;
    for (int i = 0; i < B; i++) {
      if (multinomial_prob[i] != 0) {
        nonzero_index.push_back(i);
        nonzero_prob.push_back(multinomial_prob[i]);
      }
    }
    std::discrete_distribution<int> multinomial(nonzero_prob.begin(), nonzero_prob.end());
    int cand = multinomial(generator);
    s = nonzero_index[cand];
  }
}

template <typename W>
void Graph<W>::_compute_new_rows_cols_interblock_edge_count_matrix(const std::vector<W>& M,
                                                                   const size_t& r,
                                                                   const size_t& s,
                                                                   const std::vector<size_t>& b_out,
                                                                   const std::vector<W>& count_out,
                                                                   const std::vector<size_t>& b_in,
                                                                   const std::vector<W>& count_in,
                                                                   const W& count_self,
                                                                   const bool& agg_move,
                                                                   // for return
                                                                   std::vector<W>& M_r_row,
                                                                   std::vector<W>& M_s_row,
                                                                   std::vector<W>& M_r_col,
                                                                   std::vector<W>& M_s_col) {
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
    for (size_t i = 0; i < B; i++) {
      M_r_row[i] = M[r*B + i];
      M_r_col[i] = M[i*B + r];
    }
    for (size_t i = 0; i < b_out.size(); i++) {
      M_r_row[ b_out[i] ] -= count_out[i];
    }
    W count_in_sum_r = 0;
    for (size_t i = 0; i < b_in.size(); i++) {
      if (b_in[i] == r) {
        count_in_sum_r += count_in[i];
      }
    }
    M_r_row[r] -= count_in_sum_r;
    M_r_row[s] += count_in_sum_r;
    for (size_t i = 0; i < b_in.size(); i++) {
      M_r_col[ b_in[i] ] -= count_in[i];
    }
    W count_out_sum_r = 0;
    for (size_t i = 0; i < b_out.size(); i++) {
      if (b_out[i] == r) {
        count_out_sum_r += count_out[i];
      }
    }
    M_r_col[r] -= count_out_sum_r;
    M_r_col[s] += count_out_sum_r;
  }
  M_s_row.resize(B, 0);
  M_s_col.resize(B, 0);
  for (size_t i = 0; i < B; i++) {
    M_s_row[i] = M[s*B + i];
    M_s_col[i] = M[i*B + s];
  }
  for (size_t i = 0; i < b_out.size(); i++) {
    M_s_row[ b_out[i] ] += count_out[i];
  }
  W count_in_sum_s = 0;
  for (size_t i = 0; i < b_in.size(); i++) {
    if (b_in[i] == s) {
      count_in_sum_s += count_in[i];
    }
  }
  M_s_row[r] -= count_in_sum_s;
  M_s_row[s] += count_in_sum_s;
  M_s_row[r] -= count_self;
  M_s_row[s] += count_self;

  for (size_t i = 0; i < b_in.size(); i++) {
    M_s_col[ b_in[i] ] += count_in[i];
  }

  int count_out_sum_s = 0;
  for (size_t i = 0; i < b_out.size(); i++) {
    if (b_out[i] == s) {
      count_out_sum_s += count_out[i];
    }
  }
  M_s_col[r] -= count_out_sum_s;
  M_s_col[s] += count_out_sum_s;
  M_s_col[r] -= count_self;
  M_s_col[s] += count_self;
  
} // end of compute_new_rows_cols_interblock_edge_count_matrix

template <typename W>
void Graph<W>::_compute_new_block_degree(const size_t& r,
                                         const size_t& s,
                                         const std::vector<W>& d_out,
                                         const std::vector<W>& d_in,
                                         const std::vector<W>& d,
                                         const W& k_out,
                                         const W& k_in,
                                         const W& k,
                                         // for return
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
double Graph<W>::_compute_delta_entropy(const size_t& r,
                             const size_t& s,
                             const std::vector<W>& M,
                             // TODO
                             std::vector<W> M_r_row,
                             std::vector<W> M_s_row,
                             std::vector<W> M_r_col,
                             std::vector<W> M_s_col,
                             const std::vector<W>& d_out,
                             const std::vector<W>& d_in,
                             const std::vector<W>& d_out_new,
                             const std::vector<W>& d_in_new
                             )
{
  
  size_t B = _num_blocks;
  // TODO
  std::vector<W> M_r_t1(B, 0);
  std::vector<W> M_s_t1(B, 0);
  std::vector<W> M_t2_r(B, 0);
  std::vector<W> M_t2_s(B, 0);

  for (size_t i = 0; i < B; i++) {
    M_r_t1[i] = M[r*B + i];
    M_s_t1[i] = M[s*B + i];
    M_t2_r[i] = M[i*B + r];
    M_t2_s[i] = M[i*B + s];
  }

  // remove r and s from the cols to avoid double counting
  std::vector<W> M_r_col_tmp;
  for (size_t i = 0; i < M_r_col.size(); i++) {
    if (i != r && i != s)
      M_r_col_tmp.push_back(M_r_col[i]);
  }
  M_r_col = M_r_col_tmp;

  std::vector<W> M_s_col_tmp;
  for (size_t i = 0; i < M_s_col.size(); i++) {
    if (i != r && i != s)
      M_s_col_tmp.push_back(M_s_col[i]);
  }
  M_s_col = M_s_col_tmp;

  std::vector<W> M_t2_r_tmp;
  for (size_t i = 0; i < M_t2_r.size(); i++) {
    if (i != r && i != s)
      M_t2_r_tmp.push_back(M_t2_r[i]);
  }
  M_t2_r = M_t2_r_tmp;

  std::vector<W> M_t2_s_tmp;
  for (size_t i = 0; i < M_t2_s.size(); i++) {
    if (i != r && i != s)
      M_t2_s_tmp.push_back(M_t2_s[i]);
  }
  M_t2_s = M_t2_s_tmp;


  std::vector<W> d_out_new_;
  std::vector<W> d_out_;
  for (size_t i = 0; i < d_out_new.size(); i++) {
    if (i != r && i != s)
      d_out_new_.push_back(d_out_new[i]);
  }
  for (size_t i = 0; i < d_out.size(); i++) {
    if (i != r && i != s)
      d_out_.push_back(d_out[i]);
  }

  // TODO
  // only keep non-zero entries to avoid unnecessary computation
  std::vector<W> d_in_new_r_row;
  std::vector<W> d_in_new_s_row;
  std::vector<W> M_r_row_non_zero;
  std::vector<W> M_s_row_non_zero;
  std::vector<W> d_out_new_r_col;
  std::vector<W> d_out_new_s_col;
  std::vector<W> M_r_col_non_zero;
  std::vector<W> M_s_col_non_zero;
  std::vector<W> d_in_r_t1;
  std::vector<W> d_in_s_t1;
  std::vector<W> M_r_t1_non_zero;
  std::vector<W> M_s_t1_non_zero;
  std::vector<W> d_out_r_col;
  std::vector<W> d_out_s_col;
  std::vector<W> M_t2_r_non_zero;
  std::vector<W> M_t2_s_non_zero;

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
    
  return delta_entropy;
} // end of compute_delta_entropy


template <typename W>
size_t Graph<W>::_carry_out_best_merges(const std::vector<double>& delta_entropy_for_each_block,
                                        const std::vector<int>& best_merge_for_each_block,
                                        const size_t& B_to_merge,
                                        std::vector<size_t>& b)
{
  // TODO
  size_t B = _num_blocks;

  std::vector<int> bestMerges = argsort(delta_entropy_for_each_block);
  std::vector<size_t> block_map(B);
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
  
  // TODO
  std::vector<size_t> remaining_blocks = unique(b);
  std::vector<size_t> mapping(B, -1);
  for (size_t i = 0; i < remaining_blocks.size(); i++) {
    mapping[remaining_blocks[i]] = i;
  }

  std::vector<size_t> b_return;

  for (auto& it : b) {
    b_return.push_back(mapping[it]);
  }
  b = b_return;

  return B - B_to_merge;
} // end of carry_out_best_merges

template <typename W>
double Graph<W>::_compute_overall_entropy(const std::vector<W>& M,
                                          const std::vector<W>& d_out,
                                          const std::vector<W>& d_in) 
{
  //TODO avoid this
  size_t B = _num_blocks;

  std::vector<int> nonzero_row;
  std::vector<int> nonzero_col;
  std::vector<int> edge_count_entries;
  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < B; j++) {
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

  double model_S_term = (double)B*B/_E;
  double model_S = (double)(_E * (1 + model_S_term) * log(1 + model_S_term)) - (model_S_term * log(model_S_term)) + (_N * log(B));

  return model_S + data_S;

} // end of compute_overall_entropy


template <typename W>
double Graph<W>::_compute_Hastings_correction(const std::vector<size_t>& b_out,
                                              const std::vector<W>& count_out,
                                              const std::vector<size_t>& b_in,
                                              const std::vector<W>& count_in,
                                              const size_t& s,
                                              const std::vector<W>& M,
                                              const std::vector<W>& M_r_row,
                                              const std::vector<W>& M_r_col,
                                              const std::vector<W>& d,
                                              const std::vector<W>& d_new)
{
  size_t B = _num_blocks;
  
  std::map<size_t, W> map;
  for (size_t i = 0; i < b_out.size(); i++) {
    map[b_out[i]] += count_out[i];
  }
  for (size_t i = 0; i < b_in.size(); i++) {
    map[b_in[i]] += count_in[i];
  }

  std::vector<size_t> t;
  std::vector<W> count;
  for (const auto& [key, value] : map) {
    t.push_back(key);
    count.push_back(value);
  }

  std::vector<W> M_t_s;
  std::vector<W> M_s_t;
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
} // end of compute_Hastings_correction


template <typename W>
void Graph<W>::_update_partition(const size_t& ni,
                                 const size_t& r,
                                 const size_t& s, 
                                 const std::vector<W>& M_r_row,
                                 const std::vector<W>& M_s_row,
                                 const std::vector<W>& M_r_col,
                                 const std::vector<W>& M_s_col,
                                 const std::vector<W>& d_out_new,
                                 const std::vector<W>& d_in_new,
                                 const std::vector<W>& d_new,
                                 // for return
                                 std::vector<size_t>& b,
                                 std::vector<W>& M,
                                 std::vector<W>& d_out,
                                 std::vector<W>& d_in,
                                 std::vector<W>& d)
{
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
bool Graph<W>::_prepare_for_partition_on_next_num_blocks(double S,
                                                         std::vector<size_t>& b,
                                                         std::vector<W>& M,
                                                         std::vector<W>& d,
                                                         std::vector<W>& d_out,
                                                         std::vector<W>& d_in,
                                                         size_t & B,
                                                         size_t & B_to_merge,
                                                         std::unordered_map< int, std::vector<size_t> >& old_b,
                                                         std::unordered_map< int, std::vector<W> >& old_M,
                                                         std::unordered_map< int, std::vector<W> >& old_d,
                                                         std::unordered_map< int, std::vector<W> >& old_d_out,
                                                         std::unordered_map< int, std::vector<W> >& old_d_in,
                                                         std::unordered_map< int, double >& old_S,
                                                         std::unordered_map< int, size_t >& old_B,
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
} // end of prepare_for_partition_on_next_num_blocks


} // namespace sgp

