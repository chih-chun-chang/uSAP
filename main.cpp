#include "include/propose.hpp"
#include "include/compute.hpp"
#include "include/graph.hpp"
#include "include/move.hpp"
#include "include/init.hpp"
#include "include/evaluate.hpp"

#define beta 3
#define use_sparse_matrix 0
#define num_agg_proposals_per_block 10
#define num_block_reduction_rate 0.5

#define max_num_nodal_itr 100
#define delta_entropy_threshold1 5e-4
#define delta_entropy_threshold2 1e-4
#define delta_entropy_moving_avg_window 3

int main () {

  std::string FileName("../Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar_1000_nodes");
 
  Graph g;
  g.load_graph_from_tsv(FileName);
 
  std::cout << "Number of nodes: " << g.N << std::endl;
  std::cout << "Number of edges: " << g.E << std::endl;

  int num_blocks = g.N;
  std::vector<int> partition(num_blocks);
  std::iota(partition.begin(), partition.end(), 0);


  std::vector<int> interblock_edge_count;
  std::vector<int> block_degrees_out;
  std::vector<int> block_degrees_in;
  std::vector<int> block_degrees;

  initialize_edge_counts(g.out_neighbors, 
                         num_blocks, 
                         partition, 
                         interblock_edge_count,
                         block_degrees_out,
                         block_degrees_in,
                         block_degrees);


  std::unordered_map< int, std::vector<int> > old_partition;
  std::unordered_map< int, std::vector<int> > old_interblock_edge_count;
  std::unordered_map< int, std::vector<int> > old_block_degrees;
  std::unordered_map< int, std::vector<int> > old_block_degrees_out;
  std::unordered_map< int, std::vector<int> > old_block_degrees_in;
  std::unordered_map< int, double > old_overall_entropy;
  std::unordered_map< int, int >old_num_blocks;
  
  old_overall_entropy[0] = std::numeric_limits<double>::infinity();
  old_overall_entropy[1] = std::numeric_limits<double>::infinity();
  old_overall_entropy[2] = std::numeric_limits<double>::infinity();

  bool optimal_num_blocks_found = false;

  int num_blocks_to_merge = (int)num_blocks * num_block_reduction_rate;

  while (!optimal_num_blocks_found) {
    printf("\nMerging down blocks from %d to %d\n",num_blocks, num_blocks-num_blocks_to_merge);
    std::vector<int> best_merge_for_each_block(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block(num_blocks, std::numeric_limits<double>::infinity());
    std::vector<int> block_partition(num_blocks);
    std::iota(block_partition.begin(), block_partition.end(), 0);

    for (int current_block = 0; current_block < num_blocks; current_block++) {
      for (int proposal_idx = 0; proposal_idx < num_agg_proposals_per_block; proposal_idx++) {

        std::vector< std::vector<int> > out_blocks;
        std::vector< std::vector<int> > in_blocks;
        for (int i = 0; i < num_blocks; i++) {
          if (interblock_edge_count[num_blocks*current_block + i] != 0) {
            out_blocks.push_back(std::vector<int>{i, interblock_edge_count[num_blocks*current_block + i]});
          }
          if (interblock_edge_count[num_blocks*i + current_block] != 0) {
            in_blocks.push_back(std::vector<int>{i, interblock_edge_count[num_blocks*i + current_block]});
          }
        }  

        int proposal;
        int num_out_neighbor_edges;
        int num_in_neighbor_edges;
        int num_neighbor_edges;

        propose_new_partition(current_block,
                              out_blocks,
                              in_blocks,
                              block_partition,
                              interblock_edge_count, 
                              block_degrees, 
                              num_blocks,
                              1,
                              proposal,
                              num_out_neighbor_edges,
                              num_in_neighbor_edges,
                              num_neighbor_edges);

        std::vector<int> out_blocks_0;
        std::vector<int> out_blocks_1;
        std::vector<int> in_blocks_0;
        std::vector<int> in_blocks_1;
        for (auto ele : out_blocks) {
          out_blocks_0.push_back(ele[0]);
          out_blocks_1.push_back(ele[1]);
        }
        for (auto ele : in_blocks) {
          in_blocks_0.push_back(ele[0]);
          in_blocks_1.push_back(ele[1]);
        }

        std::vector<int> new_interblock_edge_count_current_block_row;
        std::vector<int> new_interblock_edge_count_new_block_row;
        std::vector<int> new_interblock_edge_count_current_block_col;
        std::vector<int> new_interblock_edge_count_new_block_col;

        compute_new_rows_cols_interblock_edge_count_matrix(num_blocks,
                                                           interblock_edge_count,
                                                           current_block,
                                                           proposal,
                                                           out_blocks_0,
                                                           out_blocks_1,
                                                           in_blocks_0,
                                                           in_blocks_1,
                                                           interblock_edge_count[current_block*num_blocks+current_block],
                                                           1,
                                                           new_interblock_edge_count_current_block_row,
                                                           new_interblock_edge_count_new_block_row,
                                                           new_interblock_edge_count_current_block_col,
                                                           new_interblock_edge_count_new_block_col);

        std::vector<int> block_degrees_out_new;
        std::vector<int> block_degrees_in_new;
        std::vector<int> block_degrees_new;

        compute_new_block_degree(current_block,
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

        double delta_entropy = compute_delta_entropy(num_blocks,
                                                     current_block,
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
        
      }
    }
  
    num_blocks = carry_out_best_merges(delta_entropy_for_each_block,
                                       best_merge_for_each_block,
                                       partition,
                                       num_blocks,
                                       num_blocks_to_merge);
   

    initialize_edge_counts(g.out_neighbors, 
                           num_blocks, 
                           partition, 
                           interblock_edge_count,
                           block_degrees_out,
                           block_degrees_in,
                           block_degrees);

    int total_num_nodal_moves = 0;
    std::vector<double> itr_delta_entropy(max_num_nodal_itr, 0.0);

    double overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, g.N, g.E);

    printf("overall_entropy: %f\n", overall_entropy);

    for (int itr = 0; itr < max_num_nodal_itr; itr++) {

      int num_nodal_moves = 0;
      itr_delta_entropy[itr] = 0;

      for (int current_node = 0; current_node < g.N; current_node++) {
        
        int current_block = partition[current_node];
      

        int proposal;
        int num_out_neighbor_edges;
        int num_in_neighbor_edges;
        int num_neighbor_edges;

        propose_new_partition(current_block,
                              g.out_neighbors[current_node],
                              g.in_neighbors[current_node],
                              partition,
                              interblock_edge_count, 
                              block_degrees, 
                              num_blocks,
                              0,
                              proposal,
                              num_out_neighbor_edges,
                              num_in_neighbor_edges,
                              num_neighbor_edges);
        
        if (proposal != current_block) {

          std::map<int, int> blocks_out_temp;
          std::map<int, int> blocks_in_temp;
          for (auto& ele : g.out_neighbors[current_node]) {
            int key = ele[0];
            int value = ele[1];
            blocks_out_temp[partition[key]] += value;
          }
          for (auto& ele : g.in_neighbors[current_node]) {
            int key = ele[0];
            int value = ele[1];
            blocks_in_temp[partition[key]] += value;
          }

          std::vector<int> blocks_out;
          std::vector<int> count_out;
          std::vector<int> blocks_in;
          std::vector<int> count_in; 
          
          for (const auto& [key, value] : blocks_out_temp) {
            blocks_out.push_back(key);
            count_out.push_back(value);
          }
          for (const auto& [key, value] : blocks_in_temp) {
            blocks_in.push_back(key);
            count_in.push_back(value);
          }
          

          int self_edge_weight = 0;
          for (auto& ele : g.out_neighbors[current_node]) {
            if (ele[0] == current_node) self_edge_weight += ele[1];
          }
  
          std::vector<int> new_interblock_edge_count_current_block_row;
          std::vector<int> new_interblock_edge_count_new_block_row;
          std::vector<int> new_interblock_edge_count_current_block_col;
          std::vector<int> new_interblock_edge_count_new_block_col;

          compute_new_rows_cols_interblock_edge_count_matrix(num_blocks,
                                                             interblock_edge_count,
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


          std::vector<int> block_degrees_out_new;
          std::vector<int> block_degrees_in_new;
          std::vector<int> block_degrees_new;
          compute_new_block_degree(current_block,
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


          double Hastings_correction = 1;  
          if (num_neighbor_edges > 0) {
            Hastings_correction = compute_Hastings_correction(blocks_out,
                                                              count_out,
                                                              blocks_in,
                                                              count_in,
                                                              proposal,
                                                              interblock_edge_count,
                                                              new_interblock_edge_count_current_block_row,
                                                              new_interblock_edge_count_current_block_col,
                                                              num_blocks, 
                                                              block_degrees,
                                                              block_degrees_new);
          } 
    
          double delta_entropy = compute_delta_entropy(num_blocks,
                                                       current_block,
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


          double p_accept = 1;
          if (std::exp(-beta * delta_entropy)*Hastings_correction < 1) {
            p_accept = std::exp(-beta * delta_entropy)*Hastings_correction;
          }
         
          std::random_device rd; // seed the random number generator
          std::mt19937 gen(rd());
          std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
          double rand_num = uni_dist(gen);
          if ( rand_num <= p_accept) {
            total_num_nodal_moves++;
            num_nodal_moves++;
            itr_delta_entropy[itr] += delta_entropy;
            // bugs here
            update_partition(num_blocks,
                             partition, 
                             current_node, 
                             current_block, 
                             proposal, 
                             interblock_edge_count,
                             new_interblock_edge_count_current_block_row, 
                             new_interblock_edge_count_new_block_row,
                             new_interblock_edge_count_current_block_col, 
                             new_interblock_edge_count_new_block_col,
                             block_degrees_out_new, 
                             block_degrees_in_new, 
                             block_degrees_new,
                             block_degrees_out,
                             block_degrees_in,
                             block_degrees
                             );

          }
        } // endif 
      }

      double oe = overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, g.N, g.E);

      printf("Itr: %d, number of nodal moves: %d, delta S: %.5f, overall_entropy:%f \n",itr, num_nodal_moves, itr_delta_entropy[itr] / double(overall_entropy), oe);

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
    } // end outer loop

    overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, g.N, g.E);

    printf("Total number of nodal moves: %d, overall_entropy: %.2f\n", total_num_nodal_moves, overall_entropy);

    

    optimal_num_blocks_found = prepare_for_partition_on_next_num_blocks(overall_entropy,
                                                                        partition, 
                                                                        interblock_edge_count, 
                                                                        block_degrees,
                                                                        block_degrees_out, 
                                                                        block_degrees_in, 
                                                                        num_blocks,     
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
    printf("Number of blocks: [%d, %d, %d] \n", old_num_blocks[0], old_num_blocks[1], old_num_blocks[2]);

  }
  printf("partition\n");
  for (auto& b : partition) {
    printf("%d, ", b);
  }
  printf("\n");
  
  bf::evaluate(g.true_partition, partition);

  return 0;
}
