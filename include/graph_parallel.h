#ifndef GRAPH_PARALLEL_H
#define GRAPH_PARALLEL_H

#include <cuda.h>
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
#include <stack>
#include <chrono>
#include <condition_variable>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/transform.hpp>

namespace sgp {


template <unsigned i, unsigned count, bool valid = (i < count)>
struct Iterate {
  template <typename F>
  static __device__ void eval(F f){
    f(i);
    Iterate<i+1, count>::eval(f);
  }    
};   

template <unsigned i, unsigned count>
struct Iterate<i, count, false> {
  template <typename F>
  static __device__ void eval(F f) {}
};   

template <unsigned begin, unsigned end, typename F>
__device__ void _static_iterate(F&& func) {
  Iterate<0, end-begin>::eval(func);
}    

__global__ void single_task(float* sum) { 
  *sum = 0; 
}

template <size_t K, typename Input, typename W>
__global__ void compute_entropy_mat_row(
  Input M_first, Input M_last,
  Input din_first, Input din_last,
  W* dout_r, W* dout_s,
  W* dnew,
  unsigned* r, unsigned* s, float* sum) {

  unsigned B = din_last - din_first;

  __shared__ float shm[512];
  __shared__ W shm_dout_r;
  __shared__ W shm_dout_s;
  __shared__ W shm_dnew;

  // begining element of this thread in this block
  unsigned beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;

  float local_sum = 0; 
  shm[threadIdx.x] = 0; 

  shm_dout_r = *dout_r;
  shm_dout_s = *dout_s;
  shm_dnew = *dnew;

  __syncthreads();

  // unrolled version
  _static_iterate<0, K>([=, &local_sum] __device__ (int k){
    unsigned gid = beg + k*blockDim.x;
    if(gid < B && gid != *r && gid != *s){
      W Mr = *(M_first + (*r)*B + gid);
      W Ms = *(M_first + (*s)*B + gid);
      W din = *(din_first + gid);
      if (Mr != 0) { 
        local_sum += Mr * std::log((float)Mr/(din*shm_dout_r));
      }    
      if (Ms != 0) { 
        local_sum += Ms * std::log((float)Ms/(din*shm_dout_s));
      }    
      if (Mr + Ms != 0) { 
        local_sum -= (Mr + Ms) * std::log((float)(Mr + Ms)/(din*shm_dnew));
      }
    }
  });

  shm[threadIdx.x] = local_sum;

  __syncthreads();

  for(unsigned s = blockDim.x / 2; s>0; s >>= 1) {
    if(threadIdx.x < s) {
      shm[threadIdx.x] += shm[threadIdx.x + s];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    atomicAdd(sum, shm[0]);
  }
}

template <size_t K, typename Input, typename W>
__global__ void compute_entropy_mat_col(
  Input M_first, Input M_last,
  Input dout_first, Input dout_last,
  W* din_r, W* din_s,
  W* dnew,
  unsigned* r, unsigned* s, float* sum) {
  
  unsigned B = dout_last - dout_first;

  __shared__ float shm[512];
  __shared__ W shm_din_r;
  __shared__ W shm_din_s;
  __shared__ W shm_dnew;
  
  // begining element of this thread in this block
  unsigned beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;
  
  float local_sum = 0;
  shm[threadIdx.x] = 0;
  
  shm_din_r = *din_r;
  shm_din_s = *din_s;
  shm_dnew = *dnew;
  
  __syncthreads();

  // unrolled version
  _static_iterate<0, K>([=, &local_sum] __device__ (int k){
    unsigned gid = beg + k*blockDim.x;
    if(gid < B && gid != *r && gid != *s){
      W Mr = *(M_first + B*gid + *r);
      W Ms = *(M_first + B*gid + *s);
      W dout = *(dout_first + gid);
      if (Mr != 0) {
        local_sum += Mr * std::log((float)Mr/(dout*shm_din_r));
      }
      if (Ms != 0) { 
        local_sum += Ms * std::log((float)Ms/(dout*shm_din_s));
      }
      if (Mr + Ms != 0) { 
        local_sum -= (Mr + Ms) * std::log((float)(Mr + Ms)/(dout*shm_dnew));
      }
    }   
  }); 
    
  shm[threadIdx.x] = local_sum;

  __syncthreads(); 

  for(unsigned s = blockDim.x / 2; s>0; s >>= 1) {
    if(threadIdx.x < s) {
      shm[threadIdx.x] += shm[threadIdx.x + s];
    }
    __syncthreads();
  } 
    
  if(threadIdx.x == 0) {
    atomicAdd(sum, shm[0]);
  }
}   






template <typename W>
class Graph_P {

  public: 

    struct Edge {
      size_t from;
      size_t to;
      W weight;
      
      Edge(size_t from, size_t to, W weight) : from(from), to(to), weight(weight) {}
    };

    // parameters can be set by users
    int     beta                            = 3;
    size_t  block_size                      = 1024;
    size_t  num_agg_proposals_per_block     = 10; 
    float   num_block_reduction_rate        = 0.5;//0.5
    size_t  max_num_nodal_itr               = 100;
    size_t  num_batch_nodal_update          = 4;//4
    float   delta_entropy_threshold1        = 5e-4;//5e-4
    float   delta_entropy_threshold2        = 1e-4;
    size_t  delta_entropy_moving_avg_window = 3;
    bool    verbose                         = false;
    size_t  dfs_depth                       = 20;//20
    size_t  t_block_num                     = 20;

    // function used by users
    void load_graph_from_tsv(const std::string& FileName);
    const size_t& num_nodes() const { return _N; }
    const size_t& num_edges() const { return _E; }
    const std::vector<size_t>& get_partitions() const { return _P.partitions; }
    void partition();

    Graph_P(const std::string& FileName, 
      //size_t num_threads = std::thread::hardware_concurrency()) :
      size_t num_threads = 8) : 
      _executor(num_threads),
      _pt_probabilities(num_threads),
      _pt_neighbors(num_threads),
      _pt_generator(num_threads),
      _pt_newM(num_threads),
      _pt_num_nodal_move_itr(num_threads),
      _pt_delta_entropy_itr(num_threads),
      _pt_partitions_update(num_threads)
    {
      load_graph_from_tsv(FileName);
      std::random_device _rd;
      for ( auto& g : _pt_generator) {
        g.seed(_rd());
      }
    }

    // partition ground truth
    std::vector<size_t> truePartitions;

  private:
  
    struct Degree {
      std::vector<W> out;
      std::vector<W> in;
      std::vector<W> a;

      void reset(size_t B) {
        out.clear();
        in.clear();
        a.clear();
        out.resize(B);
        in.resize(B);
        a.resize(B);
      }
    };

    struct Partition {
      std::vector<size_t> partitions;
      std::vector<W> M;
      std::vector< std::vector<std::pair<size_t, W>> > Mrow;
      std::vector< std::vector<std::pair<size_t, W>> > Mcol;  
      Degree d;
      size_t B;
      size_t B_to_merge;
      float S;
    };
    
    struct OldData {
      Partition large;
      Partition med;
      Partition small;

      OldData() {
        large.B = 0;
        med.B   = 0;
        small.B = 0;
        large.S = std::numeric_limits<float>::infinity();
        med.S   = std::numeric_limits<float>::infinity();
        small.S = std::numeric_limits<float>::infinity();
      }
    };

    struct NewM {
      std::vector<W> M_r_row;
      std::vector<W> M_r_col;
      std::vector<W> M_s_row;
      std::vector<W> M_s_col;
      void reset(size_t B) {
        M_r_row.clear();
        M_r_col.clear();
        M_s_row.clear();
        M_s_col.clear();
        M_r_row.resize(B);
        M_r_col.resize(B);
        M_s_row.resize(B);
        M_s_col.resize(B);
      }
    };

    struct MergeData {
      std::vector<size_t> bestMerges;
      std::vector<size_t> remaining_blocks;
      std::set<size_t>    seen;
      std::vector<size_t> best_merge_for_each_block;
      std::vector<float>  dS_for_each_block;
      std::vector<size_t> block_map;
      std::vector<size_t> block_partitions;
      void reset (size_t B) {
        bestMerges.clear();
        remaining_blocks.clear();
        seen.clear();
        block_map.clear();
        best_merge_for_each_block.clear();
        best_merge_for_each_block.resize(B, -1);
        dS_for_each_block.clear();
        dS_for_each_block.resize(B, std::numeric_limits<float>::infinity());
        block_partitions.clear();
        block_partitions.resize(B);
        std::iota(block_partitions.begin(), block_partitions.end(), 0);
      }
    }; 

    // Graph Data
    size_t _N; // number of node
    size_t _E; // number of edge
    std::vector<Edge> _edges;
    std::vector<std::vector<std::pair<size_t, W>>> _out_neighbors;
    std::vector<std::vector<std::pair<size_t, W>>> _in_neighbors;
    std::vector<std::vector<size_t>> _neighbors;
    std::vector<std::vector<size_t>> _independent_sets;

    // taskflow
    tf::Executor _executor;

    std::vector< std::vector<float> >                _pt_probabilities;
    std::vector< std::vector<size_t> >               _pt_neighbors;
    std::vector< std::default_random_engine >        _pt_generator;
    std::vector< NewM >                              _pt_newM;
    std::vector< int >                               _pt_num_nodal_move_itr;
    std::vector< float >                             _pt_delta_entropy_itr;
    std::vector< std::vector<std::pair<size_t, W>> > _pt_partitions_update;
    
    // main thread
    OldData _old;  
    Partition _P;
    MergeData _merge_data;
    
    // functions used internally
    std::stack<int> _stack;
    std::vector<bool> _visited;

    void _scc();

    void _dfs(int v);

    void _dfs_itr();

    void _dfs_p(
      int i,
      int v,
      int b,
      std::vector<bool>& visited,
      std::vector< std::vector<size_t> > transpose
    );
    
    void _find_independent_sets();
    
    void _initialize_edge_counts();
 
    void _propose_new_partition_block(
      size_t r,
      size_t& s,
      float& dS,
      std::vector<float>& prob,
      NewM& newM,
      const std::default_random_engine& generator
    );
 
    void _propose_new_partition_nodal(
      size_t r,
      size_t ni,
      size_t& s,
      float& dS,
      float& H,
      std::vector<size_t>& neighbors,
      std::vector<float>& prob,
      NewM& newM,
      const std::default_random_engine& generator,
      int& num_nodal_move,
      float& delta_entropy_itr,
      std::vector<std::pair<size_t, W>>& partitions_update
    );

    void _carry_out_best_merges();

    float _compute_overall_entropy(
      std::vector<W>& M_r_row  
    ); 

    bool _prepare_for_partition_next();


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
    from = static_cast<size_t>(std::stoul(line.substr(start, tab_pos - start)));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    to = static_cast<size_t>(std::stoul(line.substr(start, tab_pos - start)));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    weight = static_cast<W>(std::stof(line.substr(start, tab_pos - start)));
    _edges.emplace_back(from, to, weight);
    if (from > _N) _N = from;
  }
  file.close();

  _E = _edges.size();
  
  _out_neighbors.resize(_N);
  _in_neighbors.resize(_N);
  _neighbors.resize(_N);
  for (const auto& e : _edges) {
    _out_neighbors[e.from-1].emplace_back(e.to-1, e.weight);
    _in_neighbors[e.to-1].emplace_back(e.from-1, e.weight);
    _neighbors[e.from-1].emplace_back(e.to-1);
    _neighbors[e.to-1].emplace_back(e.from-1);
  }

  // load the true partition
  std::ifstream true_file(FileName + "_truePartition.tsv");
  if (!true_file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  size_t b = 0;
  // format: node i \t block
  while (std::getline(true_file, line)) {
    size_t start = 0;
    size_t tab_pos = line.find('\t');
    size_t i = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    size_t block = std::stoi(line.substr(start, tab_pos - start));
    truePartitions.emplace_back(block-1);
    if (block > b) b = block;
  }
  true_file.close();
  
  // print
  std::cout << "Number of nodes: " << _N << std::endl;
  std::cout << "Number of edges: " << _E << std::endl;
  std::cout << "NUmber of blocks:" << b  << std::endl;

} // end of load_graph_from_tsv




template <typename W>
void Graph_P<W>::partition() {

  std::vector<W> M_r_row; // compute S
  std::vector<float> itr_delta_entropy;
  size_t nodal_batch_size = _N / num_batch_nodal_update;
  _scc();
  std::cout << _P.B << std::endl;
  _P.B_to_merge = (size_t)_P.B * num_block_reduction_rate;
  _initialize_edge_counts();
  bool optimal_B_found = false;
  while (!optimal_B_found) {
    if (verbose)  
      printf("\nMerging down blocks from %ld to %ld\n", _P.B, _P.B - _P.B_to_merge);

    _merge_data.reset(_P.B);

    // block merge
    tf::Taskflow taskflow_block;
    taskflow_block.for_each_index(0, (int)_P.B, 1, [this] (int r) {
        
        auto wid        = _executor.this_worker_id();
        auto& prob      = _pt_probabilities[wid];
        auto& generator = _pt_generator[wid];
        auto& newM      = _pt_newM[wid];

        for (size_t idx = 0; idx < num_agg_proposals_per_block; idx++) {

          size_t s;
          float dS = 0;
          
          _propose_new_partition_block(
            r, s, dS,
            prob,
            newM,
            generator
          );
          
          if (dS < _merge_data.dS_for_each_block[r]) {
            _merge_data.best_merge_for_each_block[r] = s;
            _merge_data.dS_for_each_block[r] = dS;
          }     
        } // end for proposal_idx
    }).name("block merge"); // taskflow
    _executor.run(taskflow_block).wait();

    _carry_out_best_merges();

    _initialize_edge_counts();

    int total_num_nodal_moves = 0;
    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);

    // nodal updates
    if (_P.B < _N/t_block_num) { // coarse-grained merge threshold

    bool stop = false;

    for (size_t itr = 0; itr < max_num_nodal_itr; itr++) {

      float S = _compute_overall_entropy(M_r_row);
      float mean = 0; 
      bool isf = std::isfinite(_old.large.S) && std::isfinite(_old.med.S)
        && std::isfinite(_old.small.S);
      if (itr >= (delta_entropy_moving_avg_window - 1)) {
          for (int i = itr - delta_entropy_moving_avg_window + 1; i < itr; i++) {
            mean += itr_delta_entropy[i];
          }    
          mean /= (float)(delta_entropy_moving_avg_window - 1);   
      }

      int num_nodal_moves = 0;
      std::fill(_pt_num_nodal_move_itr.begin(), _pt_num_nodal_move_itr.end(), 0);
      std::fill(_pt_delta_entropy_itr.begin(), _pt_delta_entropy_itr.end(), 0);

      for (size_t beg = 0; beg < _N; beg += nodal_batch_size) {
      
        int start = beg;
        int end = start + nodal_batch_size;
        if (end > _N) end = _N;
        
        for (auto& p_u : _pt_partitions_update) {
          p_u.clear();
        }
        tf::Taskflow taskflow_nodal;
        taskflow_nodal.for_each_index(start, end, 1, [this](int ni){
          auto wid =                _executor.this_worker_id();
          auto& prob                = _pt_probabilities[wid];
          auto& neighbors           = _pt_neighbors[wid];
          auto& generator           = _pt_generator[wid];
          auto& newM                = _pt_newM[wid];
          auto& num_nodal_move      = _pt_num_nodal_move_itr[wid];
          auto& delta_entropy_itr   = _pt_delta_entropy_itr[wid];
          auto& partitions_update   = _pt_partitions_update[wid]; 

          size_t r = _P.partitions[ni];
          size_t s;          
          float H = 1.0;
          float dS = 0;

          _propose_new_partition_nodal(
            r, ni, s, dS, H,
            neighbors,
            prob,
            newM,
            generator,
            num_nodal_move,
            delta_entropy_itr,
            partitions_update
          );

        }).name("nodal update"); // tf for each
        _executor.run(taskflow_nodal).wait();

        num_nodal_moves = std::reduce(_pt_num_nodal_move_itr.begin(), 
          _pt_num_nodal_move_itr.end(), 0.0, [](int a, int b) { return a + b; }
        );
        //itr_delta_entropy[itr] += std::reduce(_pt_delta_entropy_itr.begin(), 
        //  _pt_delta_entropy_itr.end(), 0.0, [](float a, float b) { return a + b; }
        //);
        float dS_sum = std::reduce(_pt_delta_entropy_itr.begin(), 
          _pt_delta_entropy_itr.end(), 0.0, [](float a, float b) { return a + b; }
        );
        itr_delta_entropy[itr] = dS_sum;  
        total_num_nodal_moves += num_nodal_moves;

        // update the partitions based on each thread results
        for(const auto& p_u : _pt_partitions_update) {
          for (const auto& [v, b] : p_u) {
            _P.partitions[v] = b;
          }
        } 
        _initialize_edge_counts();
       
        S+=dS_sum;
        if (itr >= (delta_entropy_moving_avg_window - 1)) {
          if (!isf) {
            if (-mean < (delta_entropy_threshold1 * S)) {
              stop = true;
              break;  
            }
          }    
          else {
            if (-mean < (delta_entropy_threshold2 * S)) {
              stop = true;
              break;
            }
          }    
        }    

      } // num_batch_nodal_update
  
      if (stop) {
        break;
      }
      
    } // end itr
    } // coarse-grained merge

    _P.S = _compute_overall_entropy(M_r_row);

    if (verbose)
      printf("Total number of nodal moves: %d, overall_entropy: %.5f\n", 
              total_num_nodal_moves, _P.S);

    optimal_B_found = _prepare_for_partition_next();

    if (verbose) {
      printf("Overall entropy: [%f, %f, %f]\n", _old.large.S, _old.med.S, _old.small.S);
      printf("Number of blocks: [%ld, %ld, %ld]\n", _old.large.B, _old.med.B, _old.small.B);
    }
  } // end while

}

template <typename W>
void Graph_P<W>::_scc(){
  _dfs_itr();

  std::vector< std::vector<size_t> > out_transpose(_N);
  for (size_t i = 0; i < _N; i++) {
    for (const auto& [v, w] : _out_neighbors[i]) {
      out_transpose[v].emplace_back(i);
    }    
  }
  
  _visited.clear();
  _visited.resize(_N, false);
  int b = 0; 
  _P.B = 0; 
  _P.partitions.resize(_N, -1); 
  while (!_stack.empty()) {
    int v = _stack.top();
    _stack.pop();
    if (!_visited[v]) {
      int i = 0; 
      std::stack<int> stack2;
      stack2.push(v);
      while (!stack2.empty() && i < dfs_depth) { // DFS depth threshold (decide B)
        int ni = stack2.top();
        stack2.pop();
        if (!_visited[ni]) {
          _visited[ni] = true;
          _P.partitions[ni] = b; 
          i++; 
          for (const auto& n : out_transpose[ni]) {
            if (!_visited[n]) {
              stack2.push(n);
            }    
          }    
        }    
      }    
      b++; 
      _P.B++;
    }    
  }

}

template <typename W>
void Graph_P<W>::_dfs(
  int v
) {

  _visited[v] = true;
  for (const auto& [n, w] : _out_neighbors[v]) {
    if (!_visited[n]) {
      _dfs(n);
    }
  }
  _stack.push(v);
  
}

template <typename W>
void Graph_P<W>::_dfs_itr() {
  
  std::vector<int> color(_N);
  std::stack<int> st;
  for (size_t v = 0; v < _N; v++) {
    if (color[v] == 0) {
      st.push(v); // visit this node
      while (!st.empty()) {
        int v = st.top();
        if (color[v] != 0) { // already seen
          st.pop();
          if (color[v] == 1) { //gray node
            _stack.push(v);
            color[v] = 2; // black, done!
          }
        }
        else {
          color[v] = 1; // gray, discover it
          for (const auto& [n, w] : _out_neighbors[v]) {
            if (color[n] == 0) {
              st.push(n);
            }
          }
        }
      }
    }
  }


}


template <typename W>
void Graph_P<W>::_dfs_p(
  int i,
  int v,
  int b,
  std::vector<bool>& visited,
  std::vector< std::vector<size_t> > transpose
) {
 
  if (i < 3) { 

  i++;  
  visited[v] = true;
  _P.partitions[v] = b;
  for (const auto& n : transpose[v]) {
    if (!visited[n]) {
      _dfs_p(i, n, b, visited, transpose);
    }
  }

  }
}




template <typename W>
void Graph_P<W>::_initialize_edge_counts() 
{
  _P.d.reset(_P.B);
  tf::Taskflow taskflow;

  if (_P.B < block_size) {
    _P.M.clear();
    _P.M.resize(_P.B*_P.B, 0);
    for (size_t node = 0; node < _out_neighbors.size(); node++) {
      if (_out_neighbors[node].size() > 0) { 
        size_t k1 = _P.partitions[node];
        for (const auto& [v, w] : _out_neighbors[node]) {
          size_t k2 = _P.partitions[v];
          _P.M[k1*_P.B + k2] += w;
        }    
      }    
    }
    for (size_t i = 0; i < _P.B; i++) {
      taskflow.emplace([this, i] () {
        for (size_t j = 0; j < _P.B; j++) {
          _P.d.out[i] += _P.M[i*_P.B + j];
          _P.d.in[i]  += _P.M[j*_P.B + i];
        }
        _P.d.a[i] = _P.d.out[i] + _P.d.in[i];
      });
    }
  }
  else {
    _P.Mrow.clear();
    _P.Mcol.clear();
    _P.Mrow.resize(_P.B);
    _P.Mcol.resize(_P.B);
    for (size_t node = 0; node < _out_neighbors.size(); node++) {
      if (_out_neighbors[node].size() > 0) { 
        size_t k1 = _P.partitions[node];
        for (const auto& [v, w] : _out_neighbors[node]) {
          size_t k2 = _P.partitions[v];
          _P.Mrow[k1].emplace_back(k2, w);
          _P.Mcol[k2].emplace_back(k1, w);
        }    
      }
    }
    for (size_t i = 0; i < _P.B; i++) {
      taskflow.emplace([this, i] () { 
        for (const auto& [v, w] : _P.Mrow[i]) {
          _P.d.out[i] += w;
        }    
        for (const auto& [v, w] : _P.Mcol[i]) {
          _P.d.in[i] += w;
        }    
        _P.d.a[i] = _P.d.out[i] + _P.d.in[i];
      });  
    }
  }
  _executor.run(taskflow).wait();
} // end of initialize_edge_counts

template <typename W>
void Graph_P<W>::_propose_new_partition_block(
  size_t r,
  size_t& s,
  float& dS,
  std::vector<float>& prob,
  NewM& newM,
  const std::default_random_engine& generator
) {
  W k_out = 0;
  W k_in = 0;
  W k = 0;

  prob.clear();
  prob.resize(_P.B);
  
  if (_P.B < block_size) {
    for (size_t i = 0; i < _P.B; i++) {
      if (_P.M[_P.B*r + i] != 0) {
        k_out += _P.M[_P.B*r + i];
        prob[i] += _P.M[_P.B*r + i]; 
      } 
      if (_P.M[_P.B*i + r] != 0) {
        k_in += _P.M[_P.B*i + r];
        prob[i] += _P.M[_P.B*i + r];
      }
    }
  }
  else {
    newM.reset(_P.B);
    for (const auto& [v, w] : _P.Mrow[r]) {
      k_out += w;
      prob[v] += w;
      newM.M_r_row[v] += w;
    }
    for (const auto& [v, w] : _P.Mcol[r]) {
      k_in += w;
      prob[v] += w;
      newM.M_r_col[v] += w;
    }
  }
  
  k = k_out + k_in;
  if ( k == 0) {
    std::uniform_int_distribution<int> randint(0, _P.B-1);
    s = randint(const_cast<std::default_random_engine&>(generator));
  }
  else {
    std::transform(prob.begin(), prob.end(), prob.begin(), 
      [k](float p){ return p/(float)k; }
    );
    std::discrete_distribution<int> dist(prob.begin(), prob.end());
    size_t rand_n = dist(const_cast<std::default_random_engine&>(generator));
    size_t u = _merge_data.block_partitions[rand_n];
    std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
    float x = uni_dist(const_cast<std::default_random_engine&>(generator));
    if ( x <= (float)_P.B/(_P.d.a[u]+_P.B) ) {
      std::uniform_int_distribution<int> choice(0, _P.B-1);
      int randIndex = choice(const_cast<std::default_random_engine&>(generator));
      if (randIndex == r) randIndex++;
      if (randIndex == _P.B) randIndex = 0;
      s = randIndex;
    }
    else {
      prob.clear();
      prob.resize(_P.B);
      if (_P.B < block_size) {
        for (size_t i = 0; i < _P.B; i++) {
          prob[i] = (float)(_P.M[u*_P.B + i] + _P.M[i*_P.B + u])/_P.d.a[u];
        }
      }
      else {
        for (const auto& [v, w] : _P.Mrow[u]) {
          prob[v] += w;
        }
        for (const auto& [v, w] : _P.Mcol[u]) {
          prob[v] += w;
        }
        std::transform(prob.begin(), prob.end(), prob.begin(), 
          [this, u](float p){ 
            return p/(float)_P.d.a[u]; 
          }
        );
      }
      float multinomial_prob_sum = std::reduce(prob.begin(), prob.end(), 0.0);
      multinomial_prob_sum -= prob[r];
      prob[r] = 0;
      if (multinomial_prob_sum == 0) {
        std::uniform_int_distribution<int> choice(0, _P.B-1);
        int randIndex = choice(const_cast<std::default_random_engine&>(generator));
        if (randIndex == r) randIndex++;
        if (randIndex == _P.B) randIndex = 0; 
        s = randIndex;
      }
      else {
        std::transform(prob.begin(), prob.end(), prob.begin(), 
          [multinomial_prob_sum](float p){ 
            return p/multinomial_prob_sum;
          }
        );
        std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
        s = multinomial(const_cast<std::default_random_engine&>(generator));
      }
    }
  }
  
  dS = 0;
  W d_out_new_r = _P.d.out[r] - k_out;
  W d_out_new_s = _P.d.out[s] + k_out;
  W d_in_new_r  = _P.d.in[r] - k_in;
  W d_in_new_s  = _P.d.in[s] + k_in;

  if (_P.B < block_size) {
    if (_P.M[r*_P.B + r] != 0) {
      dS += _P.M[r*_P.B + r] * std::log((float)_P.M[r*_P.B + r]
        /(_P.d.in[r]*_P.d.out[r]));
    }
    if (_P.M[r*_P.B + s] != 0) {
      dS += _P.M[r*_P.B + s] * std::log((float)_P.M[r*_P.B + s]
        /(_P.d.in[s]*_P.d.out[r]));
    }
    if (_P.M[s*_P.B + r] != 0) {
      dS += _P.M[s*_P.B + r] * std::log((float)_P.M[s*_P.B + r]
        /(_P.d.in[r]*_P.d.out[s]));
    }
    if (_P.M[s*_P.B + s] != 0) {
      dS += _P.M[s*_P.B + s] * std::log((float)_P.M[s*_P.B + s]
        /(_P.d.in[s]*_P.d.out[s]));
    }
    
    W c = _P.M[r*_P.B + r] + _P.M[r*_P.B + s] + _P.M[s*_P.B + r] 
      + _P.M[s*_P.B + s];
    if (c != 0) {
      dS -= c * std::log((float)c/(d_in_new_s * d_out_new_s));
    }

    /*
    for (size_t i = 0; i < _P.B; i++) {
      if (i != r && i != s) {
        // row
        if (_P.M[r*_P.B + i] != 0) {
          dS += _P.M[r*_P.B + i] * std::log((float)_P.M[r*_P.B + i]
            /(_P.d.in[i]*_P.d.out[r]));
        }
        if ( _P.M[s*_P.B + i] != 0) {
          dS +=  _P.M[s*_P.B + i] * std::log((float) _P.M[s*_P.B + i]
            /(_P.d.in[i]*_P.d.out[s]));
        }
        if (_P.M[r*_P.B + i] +_P.M[s*_P.B + i] != 0) {
          dS -= (_P.M[r*_P.B + i]+_P.M[s*_P.B + i]) * std::log((float)
            (_P.M[r*_P.B + i]+_P.M[s*_P.B + i])/(_P.d.in[i]*d_out_new_s)); 
        }
        // col
        if (_P.M[i*_P.B + r] != 0) {
          dS += _P.M[i*_P.B + r] * std::log((float)_P.M[i*_P.B + r]
            /(_P.d.out[i]*_P.d.in[r]));
        }
        if (_P.M[i*_P.B + s] != 0) {
          dS += _P.M[i*_P.B + s] * std::log((float)_P.M[i*_P.B + s]
            /(_P.d.out[i]*_P.d.in[s]));
        }
        if (_P.M[i*_P.B + r] + _P.M[i*_P.B + s] != 0) {
          dS -= (_P.M[i*_P.B + r]+_P.M[i*_P.B + s]) * std::log((float)
            (_P.M[i*_P.B + r]+_P.M[i*_P.B + s])/(_P.d.out[i]*d_in_new_s));
        }
      }
    }
    */
    const size_t K = 11;
    unsigned bs = 512;
    unsigned gs = (_P.B + bs*K - 1) / (bs*K); 


    cudaStream_t stream;
    cudaStreamCreate(&stream);
    W* gpu_M;
    W* gpu_din;
    W* gpu_dout;
    W* gpu_din_r;
    W* gpu_din_s;
    W* gpu_dout_r;
    W* gpu_dout_s;
    W* gpu_dnew_in;
    W* gpu_dnew_out;
    unsigned* gpu_r;
    unsigned* gpu_s;
    float* gpu_dS_row;
    float* gpu_dS_col;
    float dS_row;
    float dS_col;

    cudaMallocAsync(&gpu_M, sizeof(W)*_P.B*_P.B, stream);
    cudaMallocAsync(&gpu_din, sizeof(W)*_P.B, stream);
    cudaMallocAsync(&gpu_dout, sizeof(W)*_P.B, stream);
    cudaMallocAsync(&gpu_din_r, sizeof(W), stream);
    cudaMallocAsync(&gpu_din_s, sizeof(W), stream);
    cudaMallocAsync(&gpu_dout_r, sizeof(W), stream);
    cudaMallocAsync(&gpu_dout_s, sizeof(W), stream);
    cudaMallocAsync(&gpu_dnew_in, sizeof(W), stream);
    cudaMallocAsync(&gpu_dnew_out, sizeof(W), stream);
    cudaMallocAsync(&gpu_r, sizeof(unsigned), stream);
    cudaMallocAsync(&gpu_s, sizeof(unsigned), stream);
    cudaMallocAsync(&gpu_dS_row, sizeof(float), stream);
    cudaMallocAsync(&gpu_dS_col, sizeof(float), stream);

    cudaMemcpyAsync(gpu_M, _P.M.data(), sizeof(W)*_P.B*_P.B, cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_din, _P.d.in.data(), sizeof(W)*_P.B, cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_dout, _P.d.out.data(), sizeof(W)*_P.B, cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_din_r, &_P.d.in[r], sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_din_s, &_P.d.in[s], sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_dout_r, &_P.d.out[r], sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_dout_s, &_P.d.out[s], sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_dnew_in, &d_in_new_s, sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_dnew_out, &d_out_new_s, sizeof(W), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_r, &r, sizeof(unsigned), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(gpu_s, &s, sizeof(unsigned), cudaMemcpyDefault, stream);

    single_task <<< 1, 1, 0, stream >>> (gpu_dS_row);
    single_task <<< 1, 1, 0, stream >>> (gpu_dS_col);

    compute_entropy_mat_row<K> <<< gs, bs, 0, stream >>>(
      gpu_M, gpu_M+_P.B*_P.B,
      gpu_din, gpu_din+_P.B,
      gpu_dout_r, gpu_dout_s,
      gpu_dnew_out,
      gpu_r, gpu_s, gpu_dS_row
    );

    compute_entropy_mat_col<K> <<< gs, bs, 0, stream >>>(
      gpu_M, gpu_M+_P.B*_P.B,
      gpu_dout, gpu_dout+_P.B,
      gpu_din_r, gpu_din_s,
      gpu_dnew_in,
      gpu_r, gpu_s, gpu_dS_col
    );

    cudaMemcpyAsync(&dS_row, gpu_dS_row, sizeof(float), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(&dS_col, gpu_dS_col, sizeof(float), cudaMemcpyDefault, stream);

    cudaStreamSynchronize(stream);

    dS += dS_row + dS_col;

    cudaFreeAsync(gpu_M, stream);
    cudaFreeAsync(gpu_din, stream);
    cudaFreeAsync(gpu_dout, stream);
    cudaStreamDestroy(stream);

  }
  else {
    for (const auto& [v, w] : _P.Mrow[s]) {
      newM.M_s_row[v] += w;
    }
    for (const auto& [v, w] : _P.Mcol[s]) {
      newM.M_s_col[v] += w;
    }
    if (newM.M_r_row[r] != 0) {
      dS += newM.M_r_row[r] * std::log((float)newM.M_r_row[r]
        /(_P.d.in[r]*_P.d.out[r]));
    }
    if (newM.M_r_row[s] != 0) {
      dS += newM.M_r_row[s] * std::log((float)newM.M_r_row[s]
        /(_P.d.in[s]*_P.d.out[r]));
    }
    if (newM.M_s_row[r] != 0) {
      dS += newM.M_s_row[r] * std::log((float)newM.M_s_row[r]
        /(_P.d.in[r]*_P.d.out[s]));
    }
    if (newM.M_s_row[s] != 0) {
      dS += newM.M_s_row[s] * std::log((float)newM.M_s_row[s]
        /(_P.d.in[s]*_P.d.out[s]));
    }

    W c = newM.M_r_row[r] + newM.M_r_row[s] + newM.M_s_row[r] 
      + newM.M_s_row[s];
    if (c != 0) {
      dS -= c * std::log((float)c/(d_in_new_s * d_out_new_s));
    }
    for (size_t i = 0; i < _P.B; i++) {
      if (i != r && i != s) {
        // row
        if (newM.M_r_row[i] != 0) {
          dS += newM.M_r_row[i] * std::log((float)newM.M_r_row[i]
            /(_P.d.in[i]*_P.d.out[r]));
        }
        if (newM.M_s_row[i] != 0) {
          dS += newM.M_s_row[i] * std::log((float)newM.M_s_row[i]
            /(_P.d.in[i]*_P.d.out[s]));
        }
        if (newM.M_r_row[i] + newM.M_s_row[i] != 0) {
          dS -= (newM.M_r_row[i]+newM.M_s_row[i]) * std::log((float)
            (newM.M_r_row[i]+newM.M_s_row[i])/(_P.d.in[i]*d_out_new_s));
        }
        // col
        if (newM.M_r_col[i] != 0) {
          dS += newM.M_r_col[i] * std::log((float)newM.M_r_col[i]
            /(_P.d.out[i]*_P.d.in[r]));
        }
        if (newM.M_s_col[i] != 0) {
          dS += newM.M_s_col[i] * std::log((float)newM.M_s_col[i]
            /(_P.d.out[i]*_P.d.in[s]));
        }
        if (newM.M_r_col[i] + newM.M_s_col[i] != 0) {
          dS -= (newM.M_r_col[i]+newM.M_s_col[i]) * std::log((float)
            (newM.M_r_col[i]+newM.M_s_col[i])/(_P.d.out[i]*d_in_new_s));
        }
      }
    }
  }
} // 

template <typename W>
void Graph_P<W>::_propose_new_partition_nodal(
  size_t r,
  size_t ni,
  size_t& s,
  float& dS,
  float& H,
  std::vector<size_t>& neighbors,
  std::vector<float>& prob,
  NewM& newM,
  const std::default_random_engine& generator,
  int& num_nodal_move,
  float& delta_entropy_itr,
  std::vector<std::pair<size_t, W>>& partitions_update
) {

  neighbors.clear();
  prob.clear();
  W k_out = 0;
  W k_in = 0;
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
  W k = k_out + k_in;

  std::uniform_int_distribution<int> randint(0, _P.B-1);
  if (k == 0) {
    s = randint(const_cast<std::default_random_engine&>(generator));
  }
  else {
    std::transform(prob.begin(), prob.end(), prob.begin(), 
      [k](float p){
        return p/(float)k;
      }
    );
    std::discrete_distribution<int> dist(prob.begin(), prob.end());
    size_t rand_n = neighbors[dist(const_cast<std::default_random_engine&>(generator))];
    size_t u = _P.partitions[rand_n];
    std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
    float x = uni_dist(const_cast<std::default_random_engine&>(generator));
    if ( x <= (float)_P.B/(_P.d.a[u]+_P.B) ) {
      s = randint(const_cast<std::default_random_engine&>(generator));
    }
    else {
      prob.clear();
      prob.resize(_P.B);
      if (_P.B < block_size) {
        for (size_t i = 0; i < _P.B; i++) {
          prob[i] = (float)(_P.M[u*_P.B + i] + _P.M[i*_P.B + u])/_P.d.a[u];
        }
      }
      else {
        for (const auto& [v, w] : _P.Mrow[u]) {
          prob[v] += w;
        } 
        for (const auto& [v, w] : _P.Mcol[u]) {
          prob[v] += w;
        } 
        std::transform(prob.begin(), prob.end(), prob.begin(),
          [this, u](float p){
            return p/(float)_P.d.a[u];
          }
        );
      }
      std::discrete_distribution<int> multinomial(prob.begin(), prob.end());
      s = multinomial(const_cast<std::default_random_engine&>(generator));
    }
  }
  
  dS = 0;
  if ( r != s) {
    W d_out_new_r = _P.d.out[r] - k_out;
    W d_out_new_s = _P.d.out[s] + k_out;
    W d_in_new_r = _P.d.in[r] - k_in;
    W d_in_new_s = _P.d.in[s] + k_in;
    newM.reset(_P.B);
    if (_P.B < block_size) {
      for (size_t i = 0; i < _P.B; i++) {
        newM.M_r_row[i] = _P.M[r*_P.B + i];
        newM.M_s_row[i] = _P.M[s*_P.B + i];
        newM.M_r_col[i] = _P.M[i*_P.B + r];
        newM.M_s_col[i] = _P.M[i*_P.B + s];
      }
    }
    else {
      for (const auto& [v, w] : _P.Mrow[r]) {
        newM.M_r_row[v] += w; 
      }
      for (const auto& [v, w] : _P.Mrow[s]) {
        newM.M_s_row[v] += w;
      }
      for (const auto& [v, w] : _P.Mcol[r]) {
        newM.M_r_col[v] += w;
      }
      for (const auto& [v, w] : _P.Mcol[s]) {
        newM.M_s_col[v] += w;
      }
    }

    for (size_t i = 0; i < _P.B; i++) {
      if (newM.M_r_row[i] != 0) {
        dS += newM.M_r_row[i] * std::log((float)newM.M_r_row[i]
          /(_P.d.in[i]*_P.d.out[r]));
      }
      if (newM.M_s_row[i] != 0) {
        dS += newM.M_s_row[i] * std::log((float)newM.M_s_row[i]
          /(_P.d.in[i]*_P.d.out[s]));
      }
      if ( i != r && i != s) {
        if (newM.M_r_col[i] != 0) {
          dS += newM.M_r_col[i] * std::log((float)newM.M_r_col[i]
            /(_P.d.out[i]*_P.d.in[r]));
        }
        if (newM.M_s_col[i] != 0) {
          dS += newM.M_s_col[i] * std::log((float)newM.M_s_col[i]
            /(_P.d.out[i]*_P.d.in[s]));
        }
      }
    }
    
    //float p_forward = 0;
    //float p_backward = 0;
    size_t b;
    for (const auto& [v, w] : _out_neighbors[ni]) {
      b = _P.partitions[v];
      //p_forward += (float)(w * (newM.M_s_row[b] + newM.M_s_col[b] + 1))
      //  /(_P.d.a[b] + _P.B);
      if (v == ni) {
        newM.M_s_row[r] -= w;
        newM.M_s_row[s] += w;
      }
      newM.M_r_row[b] -= w;
      newM.M_s_row[b] += w;
    }
    for (const auto& [v, w] : _in_neighbors[ni]) {
      b = _P.partitions[v];
      //p_forward += (float)(w * (newM.M_s_row[b] + newM.M_s_col[b] + 1))
      //  /(_P.d.a[b] + _P.B);
      if (b == r) {
        newM.M_r_row[r] -= w;
        newM.M_r_row[s] += w;
      }
      if (b == s) {
        newM.M_s_row[r] -= w;
        newM.M_s_row[s] += w;
      }
      newM.M_r_col[b] -= w;
      newM.M_s_col[b] += w;
    }
    /*
    for (const auto& [v, w] : _out_neighbors[ni]) {
      b = _P.partitions[v];
      if (b == r) {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(d_in_new_r + d_out_new_r + _P.B);
      }
      else if (b == s) {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(d_in_new_s + d_out_new_s + _P.B);
      }
      else {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(_P.d.a[b] + _P.B);
      }
    }
    for (const auto& [v, w] : _in_neighbors[ni]) {
      b = _P.partitions[v];
      if (b == r) {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(d_in_new_r + d_out_new_r + _P.B);
      }
      else if (b == s) {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(d_in_new_s + d_out_new_s + _P.B);
      }
      else {
        p_backward += (float)(w * (newM.M_r_row[b] + newM.M_r_col[b] + 1)) 
          /(_P.d.a[b] + _P.B);
      }
    }
    H = p_backward / p_forward;
    */

    if (newM.M_r_row[r] != 0) {
      dS -= newM.M_r_row[r] * std::log((float)newM.M_r_row[r]
        /(d_in_new_r*d_out_new_r)); 
    }
    if (newM.M_r_row[s] != 0) {
      dS -= newM.M_r_row[s] * std::log((float)newM.M_r_row[s]
        /(d_in_new_s*d_out_new_r));
    }
    if (newM.M_s_row[r] != 0) {
      dS -= newM.M_s_row[r] * std::log((float)newM.M_s_row[r]
        /(d_in_new_r*d_out_new_s));
    }
    if (newM.M_s_row[s] != 0) {
      dS -= newM.M_s_row[s] * std::log((float)newM.M_s_row[s]
        /(d_in_new_s*d_out_new_s));
    }
    for (size_t i = 0; i < _P.B; i++) {
      if (i != r && i != s) {
        if (newM.M_r_row[i] != 0) {
          dS -= newM.M_r_row[i] * std::log((float)newM.M_r_row[i]
            /(_P.d.in[i]*d_out_new_r));
        }
        if (newM.M_s_row[i] != 0) {
          dS -= newM.M_s_row[i] * std::log((float)newM.M_s_row[i]
            /(_P.d.in[i]*d_out_new_s));
        }
        if (newM.M_r_col[i] != 0) {
          dS -= newM.M_r_col[i] * std::log((float)newM.M_r_col[i]
            /(_P.d.out[i]*d_in_new_r));
        }
        if (newM.M_s_col[i] != 0) {
          dS -= newM.M_s_col[i] * std::log((float)newM.M_s_col[i]
            /(_P.d.out[i]*d_in_new_s));
        }
      }
    }
    if (dS < 0) {
      num_nodal_move++;
      delta_entropy_itr += dS;
      partitions_update.emplace_back(ni, s);
    }
  } // r != s
}


template <typename W>
void Graph_P<W>::_carry_out_best_merges() {
  
  _merge_data.bestMerges.resize(_P.B);
  std::iota(_merge_data.bestMerges.begin(), _merge_data.bestMerges.end(), 0);
  std::sort(_merge_data.bestMerges.begin(), _merge_data.bestMerges.end(),
    [this](int i, int j){ 
      return _merge_data.dS_for_each_block[i] < _merge_data.dS_for_each_block[j]; 
  });
  
  _merge_data.block_map.resize(_P.B);
  std::iota(_merge_data.block_map.begin(), _merge_data.block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;
  while (num_merge < _P.B_to_merge) {
    int mergeFrom = _merge_data.bestMerges[counter];
    int mergeTo = _merge_data.block_map[
      _merge_data.best_merge_for_each_block[
      _merge_data.bestMerges[counter]
      ]];
    counter++;
    if (mergeTo != mergeFrom) {
      for (size_t i = 0; i < _P.B; i++) {
        if (_merge_data.block_map[i] == mergeFrom) 
          _merge_data.block_map[i] = mergeTo;
      }
      for (size_t i = 0; i < _P.partitions.size(); i++) {
        if (_P.partitions[i] == mergeFrom) _P.partitions[i] = mergeTo;
      }
      num_merge += 1;
    }
  }
  
  for (const auto& b : _P.partitions) {
    if (_merge_data.seen.find(b) == _merge_data.seen.end()) {
      _merge_data.seen.insert(b);
    }
  }
  _merge_data.remaining_blocks.insert(
    _merge_data.remaining_blocks.end(), _merge_data.seen.begin(), 
    _merge_data.seen.end()
  );
  std::sort(_merge_data.remaining_blocks.begin(), _merge_data.remaining_blocks.end());
  
  _merge_data.block_map.clear();
  _merge_data.block_map.resize(_P.B, -1);
  for (size_t i = 0; i < _merge_data.remaining_blocks.size(); i++) {
    _merge_data.block_map[_merge_data.remaining_blocks[i]] = i;
  }
  for (auto& it : _P.partitions) {
    it = _merge_data.block_map[it];
  }
  _P.B = _P.B - _P.B_to_merge;
} // end of carry_out_best_merges

template <typename W>
float Graph_P<W>::_compute_overall_entropy(
  std::vector<W>& M_r_row
) {

  float data_S = 0;
  if (_P.B < block_size) { 
    for (size_t i = 0; i < _P.B; i++) { 
      for (size_t j = 0; j < _P.B; j++) {
        if (_P.M[i*_P.B + j] != 0) {
          data_S -= _P.M[i*_P.B + j] * std::log((float)_P.M[i*_P.B + j]
            /(_P.d.out[i] * _P.d.in[j]));
        }
      }
    } 
  }
  else {
    M_r_row.clear();
    M_r_row.resize(_P.B);
    for (size_t i = 0; i < _P.B; i++) {
      std::fill(M_r_row.begin(), M_r_row.end(), 0);
      for (const auto& [v, w] : _P.Mrow[i]) {
        M_r_row[v] += w;
      }
      for (size_t v = 0; v < _P.B; v++) {
        if (M_r_row[v] != 0) {
          data_S -= M_r_row[v] * std::log((float)M_r_row[v]
            /(_P.d.out[i] * _P.d.in[v]));
        }
      }
    } 
  }

  float model_S_term = (float)_P.B*_P.B/_E;
  float model_S = (float)(_E * (1 + model_S_term) * std::log(1 + model_S_term)) - 
                          (model_S_term * std::log(model_S_term)) + (_N * std::log(_P.B));

  return model_S + data_S;
} // end of compute_overall_entropy


template <typename W>
bool Graph_P<W>::_prepare_for_partition_next() {
  
  bool optimal_B_found = false;

  if (_P.S <= _old.med.S) { 
    if (_old.med.B > _P.B) 
      _old.large = _old.med;
    else
      _old.small = _old.med;
    _old.med = _P;
  }
  else {
    if (_old.med.B > _P.B)
      _old.small = _P;
    else 
      _old.large = _P;
  }
 
  if (std::isinf(_old.small.S)) {
    _P = _old.med;
    _P.B_to_merge = (size_t)_P.B * num_block_reduction_rate;
    if (_P.B_to_merge == 0)  optimal_B_found = true;
  }
  else {
    if (_old.large.B - _old.small.B == 2) {
      optimal_B_found =   true;
      _P = _old.med;
    }
    else {
      if ((_old.large.B - _old.med.B) >= (_old.med.B - _old.small.B)) {  
        size_t next_B  = _old.med.B + (size_t)std::round((_old.large.B - _old.med.B) * 0.618);
        _P = _old.large;
        _P.B_to_merge = _old.large.B - next_B;
      }
      else {
        size_t next_B  = _old.small.B + (size_t)std::round((_old.med.B - _old.small.B) * 0.618);
        _P = _old.med;
        _P.B_to_merge = _old.med.B - next_B;
      }
    }
  }  
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks



} // namespace sgp

#endif // GRAPH_PARALLEL_H
