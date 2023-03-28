#include <iostream>
#include "include/graph.hpp"
#include "include/evaluate.hpp"

int main (int argc, char *argv[]) {
  
  std::string FileName("../Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar_1000_nodes");
  
  // TODO
  if(argc == 2) {
    FileName = argv[1]; 
  }
  
  sgp::Graph<int> g(FileName);

  //std::cout << g.num_nodes() << std::endl;
  //std::cout << g.num_edges() << std::endl;


  std::vector<size_t> blocks = g.partition();

  printf("partition\n");
  for (auto& b : blocks) {
    printf("%ld, ", b);
  }
  printf("\n");
  
  //bf::evaluate<size_t>(g.truePartitions, blocks);

  return 0;
}
