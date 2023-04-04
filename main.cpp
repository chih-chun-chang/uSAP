#include <iostream>
#include <chrono>
#include "include/graph.hpp"
#include "include/evaluate.hpp"

int main (int argc, char *argv[]) {
  
  std::string FileName("../Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar_1000_nodes");
  
  // TODO
  if(argc == 2) {
    FileName = argv[1]; 
  }
  
  sgp::Graph<int> g(FileName);

  std::cout << "Number of nodes: " << g.num_nodes() << std::endl;
  std::cout << "Number of edges: " << g.num_edges() << std::endl;
  g.verbose = false;

  std::cout << "Partitioning..." << std::endl;
  auto start = std::chrono::steady_clock::now();
  std::vector<size_t> blocks = g.partition();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Partitioning time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
    << " ms" << std::endl;

  std::cout << std::endl;
  bf::evaluate<size_t>(g.truePartitions, blocks);

  return 0;
}
