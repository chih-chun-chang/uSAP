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
#include <taskflow/algorithm/transform.hpp>

struct Edge {

  int from;
  int to;
  long weight;

  Edge(int from, int to, long weight) : from(from), to(to), weight(weight) {}
};

struct Graph {
  
  int N;
  int E;
  
  std::vector<Edge> edges;
  std::vector<std::vector<std::pair<int, long>>> out_neighbors;
  std::vector<std::vector<std::pair<int, long>>> in_neighbors;
  std::vector<int> truePartitions;

  Graph() : N(0), E(0) {}
};

Graph load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv");
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }

  Graph g;

  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;
  int from, to;
  long weight;
  while (std::getline(file, line)) {
    int start = 0;
    int tab_pos = line.find('\t');
    from = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    to = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    weight = static_cast<long>(std::stof(line.substr(start, tab_pos - start)));
    g.edges.emplace_back(from, to, weight);
    if (from > g.N) g.N = from;
  }
  file.close();

  g.E = g.edges.size();

  g.out_neighbors.resize(g.N);
  g.in_neighbors.resize(g.N);

  for (const auto& e : g.edges) {
    g.out_neighbors[e.from-1].emplace_back(e.to-1, e.weight);
    g.in_neighbors[e.to-1].emplace_back(e.from-1, e.weight);
  }

  // load the true partition
  std::ifstream true_file(FileName + "_truePartition.tsv");
  if (!true_file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  // format: node i \t block
  while (std::getline(true_file, line)) {
    int start = 0;
    int tab_pos = line.find('\t');
    int i = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    int block = std::stoi(line.substr(start, tab_pos - start));
    g.truePartitions.emplace_back(block-1);
  }
  true_file.close();
  
  return g;
}// end of load_graph_from_tsv


//std::vector<std::vector<int>> find_set(Graph& g) {
void find_set(Graph& g) {

  std::vector<std::vector<std::pair<int, long>>> adjList(g.out_neighbors);

  int n = 0;

  for (size_t i = 0; i < g.out_neighbors.size(); i++) {
    for (const auto& [v, w] : g.out_neighbors[i]) {
      n++;
    }   
  } 

  std::cout << n << std::endl;


}


int main (int argc, char *argv[]) {
  
  std::string FileName("./Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar");
 
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
    default:
      std::cerr << "usage: ./run [Number of Nodes=1000/5000/20000/50000]\n";
      std::exit(1);
  }

  Graph g = load_graph_from_tsv(FileName);
  std::cout << "Number of nodes: " << g.N << std::endl;
  std::cout << "Number of edges: " << g.E << std::endl;

  //std::vector<std::vector<int>> p = find_set(g);  
  find_set(g);

  return 0;
}
