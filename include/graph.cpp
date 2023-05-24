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

struct Partition {

  std::vector<long> M;
  std::vector<std::vector<std::pair<int, long>>> Mrow;
  std::vector<std::vector<std::pair<int, long>>> Mcol;
  std::vector<long> d_out;
  std::vector<long> d_in;
  std::vector<long> d;
  std::vector<int> partitions;
  int num_blocks;
  float S;

  Partition(int num_blocks) : num_blocks(num_blocks), partitions(num_blocks) {
    std::iota(partitions.begin(), partitions.end(), 0);
  }
};

struct New {
  std::vector<long> M_r_row;
  std::vector<long> M_s_row;
  std::vector<long> M_r_col;
  std::vector<long> M_s_col;
  std::vector<long> M_r_col_ori;
  std::vector<long> M_s_col_ori;
  std::vector<long> d_out;
  std::vector<long> d_in;
  std::vector<long> d;

  void reset_M(int num_blocks) {
    M_r_row.clear();
    M_s_row.clear();
    M_r_col.clear();
    M_s_col.clear();
    M_r_col_ori.clear();
    M_s_col_ori.clear();
    M_r_row.resize(num_blocks);
    M_s_row.resize(num_blocks);
    M_r_col.resize(num_blocks);
    M_s_col.resize(num_blocks);
    M_r_col_ori.resize(num_blocks);
    M_s_col_ori.resize(num_blocks);
  }
};

struct MergeData {

  std::vector<int>   best_merge_for_each_block;
  std::vector<float> delta_entropy_for_each_block;
  std::vector<int>   block_partition;
  std::vector<int>   block_map;
  std::vector<int>   bestMerges;
  std::vector<int>   remaining_blocks;
  std::set<int>      seen;

  void reset(int num_blocks) {
    best_merge_for_each_block.clear();
    best_merge_for_each_block.resize(num_blocks, -1);
    delta_entropy_for_each_block.clear();
    delta_entropy_for_each_block.resize(num_blocks, std::numeric_limits<float>::infinity());
    block_partition.clear();
    block_partition.resize(num_blocks);
    std::iota(block_partition.begin(), block_partition.end(), 0);
  }
};

struct OldData {

  Partition large;
  Partition med;
  Partition small;

  OldData() : large(0), med(0), small(0) {
    large.S = std::numeric_limits<float>::infinity();
    med.S = std::numeric_limits<float>::infinity();
    small.S = std::numeric_limits<float>::infinity();
  }
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

void initialize_edge_counts(
  const Graph& g,
  Partition& p,
  tf::Executor& executor
) {

  p.M.clear();
  p.d_out.clear();
  p.d_in.clear();
  p.d.clear();
  p.d_out.resize(p.num_blocks);
  p.d_in.resize(p.num_blocks);
  p.d.resize(p.num_blocks);

  //tf::Taskflow taskflow;

  p.M.resize(p.num_blocks * p.num_blocks);
  for (int node = 0; node < g.out_neighbors.size(); node++) {
    if (g.out_neighbors[node].size() > 0) {
      int k1 = p.partitions[node];
      for (const auto& [v, w] : g.out_neighbors[node]) {
        int k2 = p.partitions[v];
        p.M[k1*p.num_blocks + k2] += w;
      }
    }
  }
  for (int i = 0; i < p.num_blocks; i++) {
    //taskflow.emplace([i, &p](){
      for (int j = 0; j < p.num_blocks; j++) {
        p.d_out[i] += p.M[i*p.num_blocks + j];
        p.d_in[i] += p.M[j*p.num_blocks + i];
      }
      p.d[i] = p.d_out[i] + p.d_in[i];
    //});
  }

  //executor.run(taskflow).wait();


} // end of initialize

void propose_block(
  int r,
  int& s,
  float& dS,
  std::vector<float>& prob,
  New& new_,
  tf::Executor& executor,
  const std::vector<int>& partitions,
  const std::default_random_engine& generator,
  //const Partition& p
  Partition p
) {

  long k_out = 0;
  long k_in = 0;
  long k = 0;

  prob.clear();
  prob.resize(p.num_blocks);

  for (int i = 0; i < p.num_blocks; i++) {
    if (p.M[p.num_blocks*r + i] != 0) {
      k_out += p.M[p.num_blocks*r + i];
      prob[i] += (float)p.M[p.num_blocks*r + i];
    }
    if (p.M[p.num_blocks*i + r] != 0) {
      k_in += p.M[p.num_blocks*i + r];
      prob[i] += (float)p.M[p.num_blocks*i + r];
    }
  }

  k = k_out + k_in;
  std::uniform_int_distribution<int> randint(0, p.num_blocks-1);
  if ( k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  
  std::transform(prob.begin(), prob.end(), prob.begin(),
    [k](float pr){ return pr/(float)k; }
  );
  //tf::Taskflow tf;
  //tf.transform(prob.begin(), prob.end(), prob.begin(),
  //  [k](float pr){ return pr/(float)k; }
  //);
  //executor.corun(tf);

  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  int rand_n = dist(const_cast<std::default_random_engine&>(generator));
  int u = partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(generator));

  if ( x <= (float)p.num_blocks/(p.d[u]+p.num_blocks) ) {
    std::uniform_int_distribution<int> choice(0, p.num_blocks-1);
    int randIndex = choice(const_cast<std::default_random_engine&>(generator));
    if (randIndex == r) randIndex++;
    if (randIndex == p.num_blocks) randIndex = 0;
    s = randIndex;
  }
  else {
    prob.clear();
    prob.resize(p.num_blocks);
    for (int i = 0; i < p.num_blocks; i++) {
      prob[i] = (float)(p.M[u*p.num_blocks + i] + p.M[i*p.num_blocks + u])/p.d[u];
    }
    float multinomial_prob_sum = std::reduce(prob.begin(), prob.end(), 0.0);
    multinomial_prob_sum -= prob[r];
    prob[r] = 0;
    if (multinomial_prob_sum == 0) {
      std::uniform_int_distribution<int> choice(0, p.num_blocks-1);
      int randIndex = choice(const_cast<std::default_random_engine&>(generator));
      if (randIndex == r) randIndex++;
      if (randIndex == p.num_blocks) randIndex = 0;
      s = randIndex;
      return;
    }
    else {
      std::transform(prob.begin(), prob.end(), prob.begin(),
        [multinomial_prob_sum](float pr){
          return pr/multinomial_prob_sum;
        }
      );
    }
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }

  new_.d_out     = p.d_out;
  new_.d_in      = p.d_in;
  new_.d         = p.d;
  new_.d_out[r] -= k_out;
  new_.d_out[s] += k_out;
  new_.d_in[r]  -= k_in;
  new_.d_in[s]  += k_in;
  new_.d[r]     -= k;
  new_.d[s]     += k;

  // compute
  new_.M_r_row.clear();
  new_.M_r_col.clear();
  new_.M_s_row.clear();
  new_.M_s_col.clear();
  new_.M_r_col_ori.clear();
  new_.M_s_col_ori.clear();
  new_.M_s_row.resize(p.num_blocks);
  new_.M_s_col.resize(p.num_blocks);
  new_.M_r_col_ori.resize(p.num_blocks);
  new_.M_s_col_ori.resize(p.num_blocks);
  for (int i = 0; i < p.num_blocks; i++) {
    new_.M_s_row[i]     = p.M[s*p.num_blocks + i] + p.M[r*p.num_blocks + i];
    new_.M_s_col[i]     = p.M[i*p.num_blocks + s] + p.M[i*p.num_blocks + r];
    new_.M_r_col_ori[i] = p.M[i*p.num_blocks + r];
    new_.M_s_col_ori[i] = p.M[i*p.num_blocks + s];
  }
  
  new_.M_s_row[r] -= p.M[p.num_blocks*r + s];
  new_.M_s_row[s] += p.M[p.num_blocks*r + s];
  new_.M_s_row[r] -= p.M[p.num_blocks*r + r];
  new_.M_s_row[s] += p.M[p.num_blocks*r + r];
  new_.M_s_col[r] -= p.M[p.num_blocks*s + r]; // ignore
  new_.M_s_col[s] += p.M[p.num_blocks*s + r];
  new_.M_s_col[r] -= p.M[p.num_blocks*r + r];
  new_.M_s_col[s] += p.M[p.num_blocks*r + r];
  
  // dS
  dS = 0;
  //long d_out_new_r = new_.d_out[r];
  //long d_out_new_s = new_.d_out[s];
  long d_out_new_s = p.d_out[s] + k_out;
  //long d_in_new_r  = new_.d_in[r];
  //long d_in_new_s  = new_.d_in[s];
  long d_in_new_s  = p.d_in[s] + k_in;
  long d_out_r     = p.d_out[r];
  long d_out_s     = p.d_out[s];
  long d_in_r      = p.d_in[r];
  long d_in_s      = p.d_in[s];
  
  new_.M_s_col[r] = 0;
  new_.M_s_col[s] = 0;
  new_.M_r_col_ori[r] = 0;
  new_.M_r_col_ori[s] = 0;
  new_.M_s_col_ori[r] = 0;
  new_.M_s_col_ori[s] = 0;

  float M_s_row_new_dS = 0.f;
  float M_s_col_new_dS = 0.f;
  float M_r_row_ori_dS = 0.f;
  float M_s_row_ori_dS = 0.f;
  float M_r_col_ori_dS = 0.f;
  float M_s_col_ori_dS = 0.f;

  M_s_row_new_dS = std::transform_reduce(new_.M_s_row.begin(), new_.M_s_row.end(), new_.d_in.begin(), 
    0.0, std::plus<float>(), [d_out_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_s));
  });
  M_s_col_new_dS = std::transform_reduce(new_.M_s_col.begin(), new_.M_s_col.end(), new_.d_out.begin(),
    0.0, std::plus<float>(), [d_in_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_s));
  });
  M_r_row_ori_dS = std::transform_reduce(p.M.begin()+r*p.num_blocks, p.M.begin()+(r+1)*p.num_blocks, 
    p.d_in.begin(), 0.0, std::plus<float>(), [d_out_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_r));
  });
  M_s_row_ori_dS = std::transform_reduce(p.M.begin()+s*p.num_blocks, p.M.begin()+(s+1)*p.num_blocks,
    p.d_in.begin(), 0.0, std::plus<float>(), [d_out_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_s));
  });
  M_r_col_ori_dS = std::transform_reduce(new_.M_r_col_ori.begin(), new_.M_r_col_ori.end(),
    p.d_out.begin(), 0.0, std::plus<float>(), [d_in_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_r));
  });
  M_s_col_ori_dS = std::transform_reduce(new_.M_s_col_ori.begin(), new_.M_s_col_ori.end(),
    p.d_out.begin(), 0.0, std::plus<float>(), [d_in_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_s));
  });


  dS += (M_r_row_ori_dS + M_s_row_ori_dS + M_r_col_ori_dS + M_s_col_ori_dS)
      - (M_s_row_new_dS + M_s_col_new_dS);

} //end of propose nodal


/*
void compute_dS_block(
  int r,
  int s, 
  float& dS,
  //New& new_,
  New new_,
  tf::Executor& executor,
  //const Partition& p
  Partition p
) {

  new_.reset_M(p.num_blocks);
  long count_in_sum_s = 0;
  long count_out_sum_s = 0;
  long count_self = 0;

  for (int i = 0; i < p.num_blocks; i++) {
    new_.M_s_row[i]     = p.M[s*p.num_blocks + i] + p.M[r*p.num_blocks + i];
    new_.M_s_col[i]     = p.M[i*p.num_blocks + s] + p.M[i*p.num_blocks + r];
    new_.M_r_col_ori[i] = p.M[i*p.num_blocks + r];
    new_.M_s_col_ori[i] = p.M[i*p.num_blocks + s];
  }
  count_out_sum_s += p.M[p.num_blocks*r + s];
  count_in_sum_s  += p.M[p.num_blocks*s + r];
  count_self      += p.M[p.num_blocks*r + r];

  new_.M_s_row[r] -= count_in_sum_s;
  new_.M_s_row[s] += count_in_sum_s;
  new_.M_s_row[r] -= count_self;
  new_.M_s_row[s] += count_self;
  new_.M_s_col[r] -= count_out_sum_s;
  new_.M_s_col[s] += count_out_sum_s;
  new_.M_s_col[r] -= count_self;
  new_.M_s_col[s] += count_self;

  // dS
  dS = 0;
  long d_out_new_r = new_.d_out[r];
  long d_out_new_s = new_.d_out[s];
  long d_in_new_r  = new_.d_in[r];
  long d_in_new_s  = new_.d_in[s];
  long d_out_r     = p.d_out[r];
  long d_out_s     = p.d_out[s];
  long d_in_r      = p.d_in[r];
  long d_in_s      = p.d_in[s];
  
  new_.M_s_col[r] = 0;
  new_.M_s_col[s] = 0;
  new_.M_r_col_ori[r] = 0;
  new_.M_r_col_ori[s] = 0;
  new_.M_s_col_ori[r] = 0;
  new_.M_s_col_ori[s] = 0;

  float M_s_row_new_dS = 0.f;
  float M_s_col_new_dS = 0.f;
  float M_r_row_ori_dS = 0.f;
  float M_s_row_ori_dS = 0.f;
  float M_r_col_ori_dS = 0.f;
  float M_s_col_ori_dS = 0.f;

  tf::Taskflow tf;
  tf.transform_reduce(new_.M_s_row.begin(), new_.M_s_row.end(), new_.d_in.begin(), 
    M_s_row_new_dS, std::plus<float>(), [d_out_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_s));
  });
  tf.transform_reduce(new_.M_s_col.begin(), new_.M_s_col.end(), new_.d_out.begin(),
    M_s_col_new_dS, std::plus<float>(), [d_in_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_s));
  });
  tf.transform_reduce(p.M.begin()+r*p.num_blocks, p.M.begin()+(r+1)*p.num_blocks, 
    p.d_in.begin(), M_r_row_ori_dS, std::plus<float>(), [d_out_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_r));
  });
  tf.transform_reduce(p.M.begin()+s*p.num_blocks, p.M.begin()+(s+1)*p.num_blocks,
    p.d_in.begin(), M_s_row_ori_dS, std::plus<float>(), [d_out_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_s));
  });
  tf.transform_reduce(new_.M_r_col_ori.begin(), new_.M_r_col_ori.end(),
    p.d_out.begin(), M_r_col_ori_dS, std::plus<float>(), [d_in_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_r));
  });
  tf.transform_reduce(new_.M_s_col_ori.begin(), new_.M_s_col_ori.end(),
    p.d_out.begin(), M_s_col_ori_dS, std::plus<float>(), [d_in_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_s));
  });

  executor.corun(tf);
  
  //M_s_row_new_dS = std::transform_reduce(new_.M_s_row.begin(), new_.M_s_row.end(), new_.d_in.begin(), 
  //  0.0, std::plus<float>(), [d_out_new_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_s));
  //});
  //M_s_col_new_dS = std::transform_reduce(new_.M_s_col.begin(), new_.M_s_col.end(), new_.d_out.begin(),
  //  0.0, std::plus<float>(), [d_in_new_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_s));
  //});
  //M_r_row_ori_dS = std::transform_reduce(p.M.begin()+r*p.num_blocks, p.M.begin()+(r+1)*p.num_blocks, 
  //  p.d_in.begin(), 0.0, std::plus<float>(), [d_out_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_r));
  //});
  //M_s_row_ori_dS = std::transform_reduce(p.M.begin()+s*p.num_blocks, p.M.begin()+(s+1)*p.num_blocks,
  //  p.d_in.begin(), 0.0, std::plus<float>(), [d_out_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_s));
  //});
  //M_r_col_ori_dS = std::transform_reduce(new_.M_r_col_ori.begin(), new_.M_r_col_ori.end(),
  //  p.d_out.begin(), 0.0, std::plus<float>(), [d_in_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_r));
  //});
  //M_s_col_ori_dS = std::transform_reduce(new_.M_s_col_ori.begin(), new_.M_s_col_ori.end(),
  //  p.d_out.begin(), 0.0, std::plus<float>(), [d_in_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_s));
  //});

  dS += (M_r_row_ori_dS + M_s_row_ori_dS + M_r_col_ori_dS + M_s_col_ori_dS) 
        - (M_s_row_new_dS + M_s_col_new_dS);


} // end of compute_dS
*/

void carry_out_best_merges(
  int num_blocks_to_merge,
  MergeData& merge_data,
  Partition& p
) {

  merge_data.bestMerges.clear();
  merge_data.bestMerges.resize(p.num_blocks);
  std::iota(merge_data.bestMerges.begin(), merge_data.bestMerges.end(), 0);
  std::sort(merge_data.bestMerges.begin(), merge_data.bestMerges.end(),
    [&merge_data](int i, int j){ 
      return merge_data.delta_entropy_for_each_block[i] 
        < merge_data.delta_entropy_for_each_block[j]; 
  });

  merge_data.block_map.clear();
  merge_data.block_map.resize(p.num_blocks);
  std::iota(merge_data.block_map.begin(), merge_data.block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;

  while (num_merge < num_blocks_to_merge) {
    int mergeFrom = merge_data.bestMerges[counter];
    int mergeTo = 
      merge_data.block_map[
        merge_data.best_merge_for_each_block[
          merge_data.bestMerges[counter]]];
    
    counter++;
    if (mergeTo != mergeFrom) {
      for (size_t i = 0; i < p.num_blocks; i++) {
        if (merge_data.block_map[i] == mergeFrom) 
          merge_data.block_map[i] = mergeTo;
      }  
      for (size_t i = 0; i < p.partitions.size(); i++) {
        if (p.partitions[i] == mergeFrom) 
          p.partitions[i] = mergeTo;
      }    
      num_merge += 1;
    }        
  }
  
  merge_data.seen.clear();
  for (const auto& elem : p.partitions) {
    if (merge_data.seen.find(elem) == merge_data.seen.end())
      merge_data.seen.insert(elem);
  }
  merge_data.remaining_blocks.clear();
  merge_data.remaining_blocks.insert(merge_data.remaining_blocks.end(), merge_data.seen.begin(), merge_data.seen.end());
  std::sort(merge_data.remaining_blocks.begin(), merge_data.remaining_blocks.end());

  merge_data.block_map.clear();
  merge_data.block_map.resize(p.num_blocks, -1);
  for (size_t i = 0; i < merge_data.remaining_blocks.size(); i++) {
    merge_data.block_map[merge_data.remaining_blocks[i]] = i;
  }

  for (auto& it : p.partitions) {
    it = merge_data.block_map[it];
  }
  p.num_blocks = p.num_blocks - num_blocks_to_merge;
} // end of carry_out_best_merges


void propose_nodal(
  int r,
  int ni,
  int& s,
  long& k,
  std::vector<int>& neighbors,
  std::vector<float>& prob,
  New& new_,
  tf::Executor& executor,
  const std::default_random_engine& generator,
  const Partition& p,
  const Graph& g
) {
  
  long k_out = 0;
  long k_in = 0;

  neighbors.clear();
  prob.clear();

  for (const auto& [v, w] : g.out_neighbors[ni]) {
    neighbors.emplace_back(v);
    prob.emplace_back((float)w);
    k_out += w;
  }     
  for (const auto& [v, w] : g.in_neighbors[ni]) {
    neighbors.emplace_back(v);
    prob.emplace_back((float)w);
    k_in += w;
  }       
  k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, p.num_blocks-1);
  if (k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
    return;
  }
  std::transform(prob.begin(), prob.end(), prob.begin(),
    [k](float pr){
      return pr/(float)k;
    }
  );
  std::discrete_distribution<int> dist(prob.begin(), prob.end());
  size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(generator))];
  size_t u = p.partitions[rand_n];
  std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
  float x = uni_dist(const_cast<std::default_random_engine&>(generator));
  if ( x <= (float)p.num_blocks/(p.d[u]+p.num_blocks) ) {
    s = randint(const_cast<std::default_random_engine&>(generator));
  }
  else {
    prob.clear();
    prob.resize(p.num_blocks);
    for (size_t i = 0; i < p.num_blocks; i++) {
      prob[i] = (float)(p.M[u*p.num_blocks + i] + p.M[i*p.num_blocks + u])/p.d[u];
    }
    std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
    s = multinomial(const_cast<std::default_random_engine&>(generator));
  }

  new_.d_out     = p.d_out;
  new_.d_in      = p.d_in;
  new_.d         = p.d;
  new_.d_out[r] -= k_out;
  new_.d_out[s] += k_out;
  new_.d_in[r]  -= k_in;
  new_.d_in[s]  += k_in;
  new_.d[r]     -= k;
  new_.d[s]     += k;

} // end of propose nodal

void compute_dS_nodal(
  int r,
  int ni,
  int s,
  float& dS,
  New& new_,
  tf::Executor& executor,
  const Partition& p,
  const Graph& g
) {

  new_.reset_M(p.num_blocks);
  long count_in_sum_r = 0;
  long count_out_sum_r = 0;
  long count_in_sum_s = 0;
  long count_out_sum_s = 0;
  long count_self = 0;

  for (int i = 0; i < p.num_blocks; i++) {
    new_.M_r_row[i] = p.M[r*p.num_blocks + i];
    new_.M_r_col[i] = p.M[i*p.num_blocks + r];
    new_.M_s_row[i] = p.M[s*p.num_blocks + i];
    new_.M_s_col[i] = p.M[i*p.num_blocks + s];
  }
  new_.M_r_col_ori = new_.M_r_col;
  new_.M_s_col_ori = new_.M_s_col;

  // TODO: can this part done first in parallel
  size_t b;
  for (const auto& [v, w] : g.out_neighbors[ni]) {
    b = p.partitions[v];
    if (b == r) {
      count_out_sum_r += w;
    }
    if (b == s) {
      count_out_sum_s += w;
    }
    if (v == ni) {
      count_self += w;
    }
    new_.M_r_row[b] -= w;
    new_.M_s_row[b] += w;
  }
  for (const auto& [v, w] : g.in_neighbors[ni]) {
    b = p.partitions[v];
    if (b == r) {
      count_in_sum_r += w;
    }
    if (b == s) {
      count_in_sum_s += w;
    }
    new_.M_r_col[b] -= w;
    new_.M_s_col[b] += w;
  }

  new_.M_r_row[r] -= count_in_sum_r;
  new_.M_r_row[s] += count_in_sum_r;
  new_.M_r_col[r] = 0;
  new_.M_r_col[s] = 0;
  new_.M_s_row[r] -= count_in_sum_s;
  new_.M_s_row[s] += count_in_sum_s;
  new_.M_s_row[r] -= count_self;
  new_.M_s_row[s] += count_self;
  new_.M_s_col[r] = 0;
  new_.M_s_col[s] = 0;;
  new_.M_s_col[r] -= count_self;
  new_.M_s_col[s] += count_self;  
  
  // dS
  dS = 0;
  long d_out_new_r = new_.d_out[r];
  long d_out_new_s = new_.d_out[s];
  long d_in_new_r  = new_.d_in[r];
  long d_in_new_s  = new_.d_in[s];
  long d_out_r     = p.d_out[r];
  long d_out_s     = p.d_out[s];
  long d_in_r      = p.d_in[r];
  long d_in_s      = p.d_in[s];

  new_.M_r_col_ori[r] = 0;
  new_.M_r_col_ori[s] = 0;
  new_.M_s_col_ori[r] = 0;
  new_.M_s_col_ori[s] = 0;

  float M_r_row_new_dS = 0.f;
  float M_r_col_new_dS = 0.f;
  float M_s_row_new_dS = 0.f;
  float M_s_col_new_dS = 0.f;
  float M_r_row_ori_dS = 0.f;
  float M_s_row_ori_dS = 0.f;
  float M_r_col_ori_dS = 0.f;
  float M_s_col_ori_dS = 0.f;

  //tf::Taskflow tf;
  //tf.transform_reduce(new_.M_r_row.begin(), new_.M_r_row.end(), new_.d_in.begin(),
  //  M_r_row_new_dS, std::plus<float>(), [d_out_new_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_r));
  //});
  //tf.transform_reduce(new_.M_r_col.begin(), new_.M_r_col.end(), new_.d_out.begin(),
  //  M_r_col_new_dS, std::plus<float>(), [d_in_new_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_r));
  //});
  //tf.transform_reduce(new_.M_s_row.begin(), new_.M_s_row.end(), new_.d_in.begin(),
  //  M_s_row_new_dS, std::plus<float>(), [d_out_new_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_s));
  //});
  //tf.transform_reduce(new_.M_s_col.begin(), new_.M_s_col.end(), new_.d_out.begin(),
  //  M_s_col_new_dS, std::plus<float>(), [d_in_new_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_s));
  //});
  //tf.transform_reduce(p.M.begin()+r*p.num_blocks, p.M.begin()+(r+1)*p.num_blocks,
  //  p.d_in.begin(), M_r_row_ori_dS, std::plus<float>(), [d_out_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_r));
  //});
  //tf.transform_reduce(p.M.begin()+s*p.num_blocks, p.M.begin()+(s+1)*p.num_blocks,
  //  p.d_in.begin(), M_s_row_ori_dS, std::plus<float>(), [d_out_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_s));
  //});
  //tf.transform_reduce(new_.M_r_col_ori.begin(), new_.M_r_col_ori.end(),
  //  p.d_out.begin(), M_r_col_ori_dS, std::plus<float>(), [d_in_r](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_r));
  //});
  //tf.transform_reduce(new_.M_s_col_ori.begin(), new_.M_s_col_ori.end(),
  //  p.d_out.begin(), M_s_col_ori_dS, std::plus<float>(), [d_in_s](long m, long d){
  //    return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_s));
  //});

  //executor.corun(tf);
  M_r_row_new_dS = std::transform_reduce(new_.M_r_row.begin(), new_.M_r_row.end(), new_.d_in.begin(),
    0.0, std::plus<float>(), [d_out_new_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_r));
  });
  M_r_col_new_dS = std::transform_reduce(new_.M_r_col.begin(), new_.M_r_col.end(), new_.d_out.begin(),
    0.0, std::plus<float>(), [d_in_new_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_r));
  });
  M_s_row_new_dS = std::transform_reduce(new_.M_s_row.begin(), new_.M_s_row.end(), new_.d_in.begin(),
    0.0, std::plus<float>(), [d_out_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_new_s));
  });
  M_s_col_new_dS = std::transform_reduce(new_.M_s_col.begin(), new_.M_s_col.end(), new_.d_out.begin(),
    0.0, std::plus<float>(), [d_in_new_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_new_s));
  });
  M_r_row_ori_dS = std::transform_reduce(p.M.begin()+r*p.num_blocks, p.M.begin()+(r+1)*p.num_blocks,
    p.d_in.begin(), 0.0, std::plus<float>(), [d_out_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_r));
  });
  M_s_row_ori_dS = std::transform_reduce(p.M.begin()+s*p.num_blocks, p.M.begin()+(s+1)*p.num_blocks,
    p.d_in.begin(), 0.0, std::plus<float>(), [d_out_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_out_s));
  });
  M_r_col_ori_dS = std::transform_reduce(new_.M_r_col_ori.begin(), new_.M_r_col_ori.end(),
    p.d_out.begin(), 0.0, std::plus<float>(), [d_in_r](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_r));
  });
  M_s_col_ori_dS = std::transform_reduce(new_.M_s_col_ori.begin(), new_.M_s_col_ori.end(),
    p.d_out.begin(), 0.0, std::plus<float>(), [d_in_s](long m, long d){
      return m == 0 ? 0 : m * std::log(static_cast<float>(m)/(d * d_in_s));
  });


  dS += (M_r_row_ori_dS + M_s_row_ori_dS + M_r_col_ori_dS + M_s_col_ori_dS)
        - (M_r_row_new_dS + M_r_col_new_dS + M_s_row_new_dS + M_s_col_new_dS);

} // end of compute_dS_nodal

void compute_H(
  int s,
  int ni,  
  float& H,
  const New& new_,
  const Partition& p,
  const Graph& g
) {

  float p_forward = 0; 
  float p_backward = 0; 

  size_t b;
  for (const auto& [v, w] : g.out_neighbors[ni]) {
    b = p.partitions[v];
    p_forward += (float)(w * (p.M[b*p.num_blocks + s] + p.M[s*p.num_blocks + b] 
      + 1)) / (p.d[b] + p.num_blocks);
    p_backward += (float)(w * (new_.M_r_row[b] + new_.M_r_col[b] + 1)) /
      (new_.d[b] + p.num_blocks);
  }
  for (const auto& [v, w] : g.in_neighbors[ni]) {
    b = p.partitions[v];
    p_forward += (float)(w * (p.M[b*p.num_blocks + s] + p.M[s*p.num_blocks + b] 
      + 1)) / (p.d[b] + p.num_blocks);
    p_backward += (float)(w * (new_.M_r_row[b] + new_.M_r_col[b] + 1)) /
      (new_.d[b] + p.num_blocks);
  }

  H = p_backward / p_forward;

} // end of compute_H

float compute_S(
  const Partition& p,
  const Graph& g
) {   
  float data_S = 0;
  for (size_t i = 0; i < p.num_blocks; i++) {
    for (size_t j = 0; j < p.num_blocks; j++) {
      if (p.M[i*p.num_blocks + j] != 0) {
        data_S -= p.M[i*p.num_blocks + j] * std::log(p.M[i*p.num_blocks + j] /
          (float)(p.d[i] * p.d[j]));
      }   
    }   
  }

  float model_S_term = (float)p.num_blocks*p.num_blocks/g.E;
  float model_S = (float)(g.E * (1 + model_S_term) * std::log(1 + model_S_term)) -
    (model_S_term * log(model_S_term)) + (g.N * log(p.num_blocks));
  return model_S + data_S;
} // end compute_S

bool prepare_for_next(
  float S,
  float num_block_reduction_rate,
  int& num_blocks_to_merge,
  OldData& old,
  Partition& p
) {
  
  bool optimal = false;
  p.S = S;
  if (S <= old.med.S) {
    if (old.med.num_blocks > p.num_blocks) {
      old.large = old.med;
    }
    else {
      old.small = old.med;
    }
    old.med = p;
  }
  else {
    if (old.med.num_blocks > p.num_blocks) {
      old.small = p;
    }
    else {
      old.large = p;
    }
  } 
  if (std::isinf(old.small.S)) {
    num_blocks_to_merge = (int)p.num_blocks * num_block_reduction_rate;
    if (num_blocks_to_merge == 0) optimal = true;
    p = old.med;
  }
  else {
    if (old.large.num_blocks - old.small.num_blocks == 2) {
      optimal = true;
      p = old.med;
    }
    else {
      if ((old.large.num_blocks - old.med.num_blocks) >=
          (old.med.num_blocks - old.small.num_blocks)) {
        int next_B_to_try = old.med.num_blocks +
          static_cast<int>(std::round((old.large.num_blocks - old.med.num_blocks) * 0.618));
        num_blocks_to_merge = old.large.num_blocks - next_B_to_try;
        p = old.large;
      }
      else {
        int next_B_to_try = old.small.num_blocks + 
          static_cast<int>(std::round((old.med.num_blocks - old.small.num_blocks) * 0.618));
        num_blocks_to_merge = old.med.num_blocks - next_B_to_try;
        p = old.med;
      }
    }
  }
  return optimal;
} // end prepare

std::vector<int> partition(const Graph& g) {

  // params
  int     beta                            = 3;
  int     block_size                      = 1024;
  int     num_agg_proposals_per_block     = 10;
  float   num_block_reduction_rate        = 0.5;
  int     max_num_nodal_itr               = 100;
  int     num_batch_nodal_update          = 4;
  float   delta_entropy_threshold1        = 5e-4;
  float   delta_entropy_threshold2        = 1e-4;
  int     delta_entropy_moving_avg_window = 3;
  int     num_blocks                      = g.N;
  int     num_blocks_to_merge             = (int)num_blocks * num_block_reduction_rate; 
  int     nodal_update_batch_size         = g.N / num_batch_nodal_update;
  // global shared variable
  MergeData merge_data;
  std::vector<float> itr_delta_entropy;
  OldData old;

  // Taskflow
  tf::Executor executor;

  // init
  Partition p(num_blocks);
  initialize_edge_counts(g, p, executor);

  // thread local
  int num_threads = std::thread::hardware_concurrency();
  std::vector<std::vector<float>> pt_probs(num_threads);
  std::vector<New> pt_new(num_threads);
  std::vector<std::default_random_engine> pt_generator(num_threads);
  std::random_device rd;
  for (auto& gen : pt_generator) {
    gen.seed(rd());
  }  
  std::vector<int> pt_num_nodal_move_itr(num_threads);
  std::vector<float> pt_delta_entropy_itr(num_threads);
  std::vector<std::vector<std::pair<int, long>>> pt_partitions(num_threads);
  std::vector<std::vector<int>> pt_neighbors(num_threads);

  std::vector<Partition> pt_par(num_threads, Partition(p.num_blocks));


  bool found = false;
  //while (!found) {
  for (int ii = 0; found == false; ii++) {


  // init for this merge
  //
  std::cout << "perform block merge\n";
  merge_data.reset(p.num_blocks);


  // reset all the thread data
  //pt_probs.clear();
  //pt_probs.resize(num_threads);
  pt_new.clear();
  pt_new.resize(num_threads);
  for (auto& pt_p : pt_par) {
    pt_p = p;
  }   

  
  tf::Taskflow taskflow;
  // perform a fake block merge 
  taskflow.for_each_index(0, p.num_blocks, 1, [&](int r){
  //for (int r = 0; r < p.num_blocks; r++) {
    // get the thread local
    auto wid        = executor.this_worker_id();
    //int wid = 0;
    auto& prob      = pt_probs[wid];
    auto& generator = pt_generator[wid];
    auto& new_      = pt_new[wid];
    
    auto& pt_p = pt_par[wid];
    
    
    // num proposal for each block
    for (size_t _=0; _<num_agg_proposals_per_block; _++) {
      int s;
      float dS;
      propose_block(r, s, dS, prob, new_, executor, merge_data.block_partition, generator, pt_p);
      //compute_dS_block(r, s, dS, new_, executor, pt_p);
      if (dS < merge_data.delta_entropy_for_each_block[r]) {
        merge_data.best_merge_for_each_block[r] = s;
        merge_data.delta_entropy_for_each_block[r] = dS;
      } // end if
    }   // end propose idx       
  });   // end tf::for_each
  executor.run(taskflow).wait();
  //}

  // perform merge based on the proposal
  carry_out_best_merges(num_blocks_to_merge, merge_data, p);

  initialize_edge_counts(g, p, executor);

  int total_num_nodal_moves = 0;
  itr_delta_entropy.clear();
  itr_delta_entropy.resize(max_num_nodal_itr, 0.0);


  std::cout << "perform nodal update\n";
  // perform nodal update
  for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {
    int num_nodal_moves = 0;
    std::fill(pt_num_nodal_move_itr.begin(), pt_num_nodal_move_itr.end(), 0);
    std::fill(pt_delta_entropy_itr.begin(), pt_delta_entropy_itr.end(), 0);
    for (int beg = 0; beg < g.N; beg += nodal_update_batch_size ) {
      int end = beg + nodal_update_batch_size ;
      if (end > g.N) 
        end = g.N;
      for (auto& pt_par : pt_partitions) {
        pt_par.clear();
      }
      tf::Taskflow taskflow2;
      taskflow2.for_each_index(beg, end, 1, [&](int ni){
      //for ( int ni = beg; ni < end; ni++) {
        auto wid                = executor.this_worker_id();
        //int wid = 0;
        auto& neighbors         = pt_neighbors[wid];
        auto& prob              = pt_probs[wid];
        auto& generator         = pt_generator[wid];
        auto& new_              = pt_new[wid];
        auto& num_nodal_move    = pt_num_nodal_move_itr[wid];
        auto& dS_itr            = pt_delta_entropy_itr[wid];
        auto& partitions_update = pt_partitions[wid];

        int r = p.partitions[ni];
        int s;
        long k;
        float dS;
        float H = 1.0;
        propose_nodal(r, ni, s, k, neighbors, prob, new_, executor, generator, p, g);
        if (r != s) {
          compute_dS_nodal(r, ni, s, dS, new_, executor, p, g);
          if ( k != 0 ) {
            compute_H(s, ni, H, new_, p, g);
          } 
          float p_accept = std::min(static_cast<float>(std::exp(-beta * dS)) * H, 1.0f);
          std::uniform_real_distribution<float> uni_dist(0.0, 1.0); // can I do thread
          if ( uni_dist(generator) <= p_accept) {
            num_nodal_move++;
            dS_itr += dS;
            partitions_update.emplace_back(ni, s);
          } // accept or not        
        }   // r != s
      });   // end tf::for_each
      executor.run(taskflow2).wait();
      //}
      num_nodal_moves = std::reduce(pt_num_nodal_move_itr.begin(), pt_num_nodal_move_itr.end(), 0,
        [](int a, int b) { return a + b; }
      );
      itr_delta_entropy[itr] = std::reduce(pt_delta_entropy_itr.begin(), pt_delta_entropy_itr.end(), 0.0,
        [](float a, float b) { return a + b; }
      );
      total_num_nodal_moves += num_nodal_moves;
      for(const auto& par : pt_partitions) { //// TODO: I think this can also be parallelized
        for (const auto& [v, b] : par) {
          p.partitions[v] = b;
        }
      }
      initialize_edge_counts(g, p, executor);
    } // end batch nodal update
    float S = compute_S(p, g);
    if (itr >= (delta_entropy_moving_avg_window - 1)) {    
      bool isf = std::isfinite(old.large.S) && 
        std::isfinite(old.med.S) && std::isfinite(old.small.S);
      float mean = 0;
      for (int i = itr - delta_entropy_moving_avg_window + 1; i < itr; i++) {
        mean += itr_delta_entropy[i];
      }
      mean /= (float)(delta_entropy_moving_avg_window - 1);
      if (!isf) {
        if (-mean < (delta_entropy_threshold1 * S)) break;
      }
      else {
        if (-mean < (delta_entropy_threshold2 * S)) break;
      }
    } // end if
  } // end nodal itr
  float S = compute_S(p, g);
  found = prepare_for_next(S, num_block_reduction_rate,
    num_blocks_to_merge, old, p);
  
  printf("Overall entropy: [%f, %f, %f] \n", old.large.S, old.med.S, old.small.S);
  printf("Number of blocks: [%d, %d, %d] \n",old.large.num_blocks, old.med.num_blocks, old.small.num_blocks);
  
  
  } //while
  
  
  //printf("Overall entropy: [%f, %f, %f] \n", old.large.S, old.med.S, old.small.S);
  //printf("Number of blocks: [%d, %d, %d] \n",old.large.num_blocks, old.med.num_blocks, old.small.num_blocks);
  std::cout << old.med.num_blocks << std::endl;
  return p.partitions;
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

  std::vector<int> p = partition(g);  


  return 0;
}
