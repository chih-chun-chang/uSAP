#include <iostream>
#include <chrono>
#include "include/graph.hpp"
#include "include/graph_parallel.hpp"
#include "include/evaluate.hpp"

int main (int argc, char *argv[]) {
  
  std::string FileName("../Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar");
 
  if(argc != 2) {
    std::cerr << "usage: ./run [Number of Nodes]\n";
    std::exit(1);
  }

  int num_nodes = std::stoi(argv[1]);

  switch(num_nodes)  {
    case 1000:
      FileName += "_1000_nodes";
      break;
    case 5000:
      FileName += "_5000_nodes";
      break;
    case 20000:
      FileName += "_20000_nodes";
      break;
    case 50000:
      FileName += "_50000_nodes";
      break;
    default:
      std::cerr << "usage: ./run [Number of Nodes=1000/5000/20000/50000]\n";
      std::exit(1);
  }

  //sgp::Graph<int> g(FileName);

  //std::cout << "Number of nodes: " << g.num_nodes() << std::endl;
  //std::cout << "Number of edges: " << g.num_edges() << std::endl;
  //g.verbose = false;

  //std::cout << "Partitioning..." << std::endl;
  //auto start = std::chrono::steady_clock::now();
  //std::vector<size_t> blocks = g.partition();
  //auto end = std::chrono::steady_clock::now();
  //std::cout << "Partitioning time: " << 
  //  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
  //  << " ms" << std::endl;

  //std::cout << std::endl;
  ////bf::evaluate<size_t>(g.truePartitions, blocks);

  // parallel
  sgp::Graph_P<int> g_p(FileName);
  std::cout << "Number of nodes: " << g_p.num_nodes() << std::endl;
  std::cout << "Number of edges: " << g_p.num_edges() << std::endl;
  g_p.verbose = false;

  std::cout << "Partitioning..." << std::endl;
  auto start = std::chrono::steady_clock::now();
  std::vector<size_t> blocks2 = g_p.partition();
  auto end = std::chrono::steady_clock::now();
  std::cout << "(Parallel) Partitioning time: " <<  
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
    << " ms" << std::endl;

  std::cout << std::endl;
  //bf::evaluate<size_t>(g_p.truePartitions, blocks2);
  
  return 0;
}
