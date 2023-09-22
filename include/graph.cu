#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <set>

//const int TILE_DIM = 32;
//const int BLOCK_ROWS = 8;

template <typename Node, typename Weight>
struct Edge {
  Node   s;
  Node   t;
  Weight w;
  Edge(Node s, Node t, Weight w) : s(s), t(t), w(w) {}
};

template <typename Node, typename Weight>
struct Csr {
  std::vector<unsigned> adj_ptr;
  std::vector<Node> adj_node;
  std::vector<Weight> adj_wgt;
  std::vector<Weight> deg;
};

template <typename Node, typename Weight>
struct Graph {
  long N; // number of node and edge
  long E;
  std::vector<Edge<Node, Weight>> edges;
  Csr<Node, Weight> csr_out;
  Csr<Node, Weight> csr_in;
  std::vector<Weight> deg;
  std::vector<Node> blocks;
  Graph() : N(0), E(0) {}
};

template <typename Node, typename Weight>
struct Block {
  Csr<Node, Weight> csr_out;
  Csr<Node, Weight> csr_in;
  std::vector<Weight> deg;
};

template <typename Node, typename Weight>
bool compare_s(const Edge<Node, Weight>& e1, const Edge<Node, Weight>& e2) {
  return e1.s < e2.s;
}

template <typename Node, typename Weight>
bool compare_t(const Edge<Node, Weight>& e1, const Edge<Node, Weight>& e2) {
  return e1.t < e2.t;
}

template <typename Node, typename Weight>
Graph<Node, Weight> load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv");
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;

  Graph<Node, Weight> g;
  Node s, t;
  Weight w;
  unsigned start, tab_pos;
  while (std::getline(file, line)) {
    start = 0;
    tab_pos = line.find('\t');
    s = static_cast<Node>(std::stoi(line.substr(start, tab_pos - start)));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    t = static_cast<Node>(std::stoi(line.substr(start, tab_pos - start)));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    w = static_cast<Weight>(std::stof(line.substr(start, tab_pos - start)));
    g.edges.emplace_back(s-1, t-1, w);
    if (s > g.N) g.N = s; 
  }
  file.close();

  g.E = g.edges.size();
  g.deg.resize(g.N);
  
  std::sort(g.edges.begin(), g.edges.end(), compare_s<Node, Weight>);
  g.csr_out.adj_ptr.emplace_back(0);
  g.csr_out.deg.resize(g.N);
  s = 0;
  for (unsigned i = 0; i < g.E; i++) {
    if (g.edges[i].s != s) {
      s++;
      g.csr_out.adj_ptr.emplace_back(i);
    }
    g.csr_out.adj_node.emplace_back(g.edges[i].t);
    g.csr_out.adj_wgt.emplace_back(g.edges[i].w);
    g.csr_out.deg[g.edges[i].s] += g.edges[i].w;
    g.deg[g.edges[i].s] += g.edges[i].w;
  }

  std::sort(g.edges.begin(), g.edges.end(), compare_t<Node, Weight>);
  g.csr_in.adj_ptr.emplace_back(0);
  g.csr_in.deg.resize(g.N);
  t = 0;
  for (unsigned i = 0; i < g.E; i++) {
    if (g.edges[i].t != t) {
      t++;
      g.csr_in.adj_ptr.emplace_back(i);
    }
    g.csr_in.adj_node.emplace_back(g.edges[i].s);
    g.csr_in.adj_wgt.emplace_back(g.edges[i].w);
    g.csr_in.deg[g.edges[i].t] += g.edges[i].w;
    g.deg[g.edges[i].t] += g.edges[i].w;
  }
  return g;
}


// ----------------------CUDA kernel ---------------------//

template <typename Node, typename Weight>
__global__ void random_block_generator(Node* gpu_random_blocks, unsigned B) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_random_blocks[idx] = curand(&state) % B;
  }

}

template <typename Node, typename Weight>
__global__ void random_block_generator_nodal(Node* gpu_random_blocks, unsigned B, unsigned N) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_random_blocks[idx] = curand(&state) % B;
  }

}


template <typename Node, typename Weight>
__global__ void uniform_number_generator(float* gpu_uniform_x, unsigned B) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_uniform_x[idx] = curand_uniform(&state);
  }

}

template <typename Node, typename Weight>
__global__ void calculate_acceptance_prob(float* gpu_acceptance_prob, Weight*
                                          deg, unsigned B) {

  unsigned idx =  blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    gpu_acceptance_prob[idx] = (float)B/(deg[idx]+B);
  }

}

template <typename Node, typename Weight>
__global__ void sample_neighbors(Node* gpu_sampling_neighbors, 
                                 unsigned* csr_out_adj_ptr,
  				 Node* csr_out_adj_node,
                                 Weight* csr_out_adj_wgt,
                                 unsigned* csr_in_adj_ptr,
                                 Node* csr_in_adj_node,
                                 Weight* csr_in_adj_wgt,
                                 Weight* deg, unsigned B) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    float random = curand_uniform(&state);

    unsigned out_s, out_e;
    unsigned in_s, in_e;
    out_s = csr_out_adj_ptr[idx];
    in_s = csr_in_adj_ptr[idx];
    if (idx + 1 < B) {
      out_e = csr_out_adj_ptr[idx+1];
      in_e = csr_in_adj_ptr[idx+1];
    }
    else {
      out_e = B;
      in_e = B; 
    }

    float prob_sum = 0.0f;
    unsigned neighbor;
    unsigned find = 0;
    for (unsigned i = out_s; i < out_e; i++) {
      prob_sum += (float)csr_out_adj_wgt[i]/deg[idx];
      if (random > prob_sum) {
        neighbor = csr_out_adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (unsigned i = in_s; i < in_e; i++) {
        prob_sum += (float)csr_in_adj_wgt[i]/deg[idx];
        if (random > prob_sum) {
          neighbor = csr_in_adj_node[i];
          break;
        }
      }
    }

    gpu_sampling_neighbors[idx] = neighbor;
     
  }

}


template <typename Node, typename Weight>
__global__ void sample_neighbors_nodal(Node* gpu_sampling_neighbors,
                                       unsigned* csr_out_adj_ptr,
                                       Node* csr_out_adj_node,
                                       Weight* csr_out_adj_wgt,
                                       unsigned* csr_in_adj_ptr,
                                       Node* csr_in_adj_node,
                                       Weight* csr_in_adj_wgt,
                                       Weight* deg, 
				       Node* blocks, unsigned N) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    float random = curand_uniform(&state);

    unsigned out_s, out_e;
    unsigned in_s, in_e;

    out_s = csr_out_adj_ptr[idx];
    in_s = csr_in_adj_ptr[idx];
    if (idx + 1 < N) {
      out_e = csr_out_adj_ptr[idx+1];
      in_e = csr_in_adj_ptr[idx+1];
    }
    else {
      out_e = N;
      in_e = N;
    }

    float prob_sum = 0.0f;
    unsigned neighbor;
    unsigned find = 0;
    for (unsigned i = out_s; i < out_e; i++) {
      prob_sum += (float)csr_out_adj_wgt[i]/deg[idx];
      if (random > prob_sum) {
        neighbor = blocks[csr_out_adj_node[i]];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (unsigned i = in_s; i < in_e; i++) {
        prob_sum += (float)csr_in_adj_wgt[i]/deg[idx];
        if (random > prob_sum) {
          neighbor = blocks[csr_in_adj_node[i]];
          break;
        }
      }
    }

    gpu_sampling_neighbors[idx] = neighbor;
  
  }

}



template <typename Node, typename Weight>
__global__ void calculate_dS_out(float* dS_out,
	       			 unsigned* csr_out_adj_ptr,
  				 Node* csr_out_adj_node,
  				 Weight* csr_out_adj_wgt,
  				 Weight* csr_out_deg,
  				 Weight* csr_in_deg,
                                 unsigned B) {
  
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    unsigned out_s = csr_out_adj_ptr[idx];
    unsigned out_e;
    if (idx + 1 < B) {
      out_e = csr_out_adj_ptr[idx+1];
    }
    else {
      out_e = B;
    }
    float dS = 0;
    for (unsigned i = out_s; i < out_e; i++) {
      dS += (float)csr_out_adj_wgt[i] * std::log((float)csr_out_adj_wgt[i]
          / (csr_out_deg[idx] * csr_in_deg[csr_out_adj_node[i]]));
    }
    dS_out[idx] = dS;
  }

}

template <typename Node, typename Weight>
__global__ void calculate_dS_in(float* dS_in, 
                                unsigned* csr_in_adj_ptr,
                                Node* csr_in_adj_node,
                                Weight* csr_in_adj_wgt,
                                Weight* csr_in_deg,
                                Weight* csr_out_deg,
				unsigned B) {
    
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    unsigned in_s = csr_in_adj_ptr[idx];
    unsigned in_e;
    if (idx + 1 < B) {
      in_e = csr_in_adj_ptr[idx+1];
    }
    else {
      in_e = B;
    }
    float dS = 0;
    for (unsigned i = in_s; i < in_e; i++) {
      dS += (float)csr_in_adj_wgt[i] * std::log((float)csr_in_adj_wgt[i]
          / (csr_out_deg[csr_in_adj_node[i]] * csr_in_deg[idx]));
    }
    dS_in[idx] = dS;
  }

}


// Potential Bottleneck?
template <typename Node, typename Weight>
__global__ void calculate_dS_new_out(float* dS_new_out, Node* s,
                                     unsigned* csr_out_adj_ptr,
                                     Node* csr_out_adj_node,
                                     Weight* csr_out_adj_wgt,
                                     Weight* csr_out_deg,
                                     Weight* csr_in_deg, 
				     unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    unsigned s_ = s[r]; 
    unsigned out_r_s, out_r_e;
    unsigned out_s_s, out_s_e;
    out_r_s = csr_out_adj_ptr[r];
    out_s_s = csr_out_adj_ptr[s_];
    if (r + 1 < B) {
      out_r_e = csr_out_adj_ptr[r+1];
    }   
    else {
      out_r_e = B;
    }    
    if (s_ + 1 < B) {
      out_s_e = csr_out_adj_ptr[s_+1];
    }
    else {
      out_s_e = B;
    }
    unsigned i = out_r_s, j = out_s_s;
    float dS = 0;
    Weight w;
    Weight dout = csr_out_deg[r] + csr_out_deg[s_];
    Node n;
    // ascending order
    while (i < out_r_e && j < out_s_e) {
      if (csr_out_adj_node[i] < csr_out_adj_node[j]) {
        w = csr_out_adj_wgt[i];
        n = csr_out_adj_node[i];
        i++;
      }
      else if (csr_out_adj_node[i] > csr_out_adj_node[j]) {
        w = csr_out_adj_wgt[j];
        n = csr_out_adj_node[j];
        j++;
      }
      else {
        w = csr_out_adj_wgt[i] + csr_out_adj_wgt[j];
        n = csr_out_adj_node[i];
        i++;
        j++;
      }
      dS -= w * std::log((float)w/(dout*csr_in_deg[n]));
    }
    for (; i < out_r_e; i++) {
      w = csr_out_adj_wgt[i];
      n = csr_out_adj_node[i];
      dS -= w * std::log((float)w/(dout*csr_in_deg[n]));
    }
    for (; j < out_s_e; j++) {
      w = csr_out_adj_wgt[j];
      n = csr_out_adj_node[j];
      dS -= w * std::log((float)w/(dout*csr_out_deg[n]));
    }
    dS_new_out[r] = dS;
  }

}


template <typename Node, typename Weight>
__global__ void calculate_dS_new_in(float* dS_new_in, Node* s,
                                    unsigned* csr_in_adj_ptr,
                                    Node* csr_in_adj_node,
                                    Weight* csr_in_adj_wgt,
                                    Weight* csr_in_deg,                    
				    Weight* csr_out_deg,
				    unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    unsigned s_ = s[r];
    unsigned in_r_s, in_r_e;
    unsigned in_s_s, in_s_e;
    in_r_s = csr_in_adj_ptr[r];
    in_s_s = csr_in_adj_ptr[s_];
    if (r + 1 < B) {
      in_r_e = csr_in_adj_ptr[r+1];
    } 
    else {
      in_r_e = B;
    } 
    if (s_ + 1 < B) {
      in_s_e = csr_in_adj_ptr[s_+1];
    }
    else {
      in_s_e = B;
    }
    unsigned i = in_r_s, j = in_s_s;
    float dS = 0;
    Weight w;
    Weight din = csr_in_deg[r] + csr_in_deg[s_];
    Node n;
    // ascending order
    while (i < in_r_e && j < in_s_e) {
      if (csr_in_adj_node[i] < csr_in_adj_node[j]) {
        w = csr_in_adj_wgt[i];
        n = csr_in_adj_node[i];
        i++;
      }
      else if (csr_in_adj_node[i] > csr_in_adj_node[j]) {
        w = csr_in_adj_wgt[j];
        n = csr_in_adj_node[j];
        j++;
      }
      else {
        w = csr_in_adj_wgt[i] + csr_in_adj_wgt[j];
        n = csr_in_adj_node[i];
        i++;
        j++;
      }
      dS -= w * std::log((float)w/(csr_out_deg[n]*din));
    }
    for (; i < in_r_e; i++) {
      w = csr_in_adj_wgt[i];
      n = csr_in_adj_node[i];
      dS -= w * std::log((float)w/(csr_out_deg[n]*din));
    }
    for (; j < in_s_e; j++) {
      w = csr_in_adj_wgt[j];
      n = csr_in_adj_node[j];
      dS -= w * std::log((float)w/(csr_out_deg[n]*din));
    }
    dS_new_in[r] = dS;
  }

}

template <typename Node, typename Weight>
__global__ void calculate_dS_overall(float* dS, float* dS_out, float* dS_in,
                                     float* dS_new_out, float* dS_new_in, 
				     unsigned* csr_out_adj_ptr,
                                     Node* csr_out_adj_node,
                                     Weight* csr_out_adj_wgt,
                                     Weight* csr_out_deg,
                                     Weight* csr_in_deg,
				     Node* s,
                                     Node* bestS,
				     unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    unsigned s_ = s[r];
    float dS_ = 0;
    dS_ += dS_out[r];
    dS_ += dS_out[s_];
    dS_ += dS_in[r];
    dS_ += dS_in[s_];
    dS_ -= dS_new_out[r];
    dS_ -= dS_new_in[r];
    
    unsigned out_r_s = csr_out_adj_ptr[r];
    unsigned out_r_e;
    unsigned out_s_s = csr_out_adj_ptr[s_];
    unsigned out_s_e;
    if (r + 1 < B) {
      out_r_e = csr_out_adj_ptr[r+1];
    } 
    else {
      out_r_e = B;
    }
    if (s_ + 1 < B) {
      out_s_e = csr_out_adj_ptr[s_+1];
    }
    else {
      out_s_e = B;
    }
    for (unsigned i = out_r_s; i < out_r_e; i++) {
      if (csr_out_adj_node[i] == r) {
        dS_ -= csr_out_adj_wgt[i]
          * std::log((float)csr_out_adj_wgt[i]/(csr_out_deg[r]*csr_in_deg[r]));
      }
      if (csr_out_adj_node[i] == s_) {
        dS_ -= csr_out_adj_wgt[i]
          * std::log((float)csr_out_adj_wgt[i]/(csr_out_deg[r]*csr_in_deg[s_]));
      }
    }
    for (unsigned i = out_s_s; i < out_s_e; i++) {
      if (csr_out_adj_node[i] == r) {
        dS_ -= csr_out_adj_wgt[i]
          * std::log((float)csr_out_adj_wgt[i]/(csr_out_deg[s_]*csr_in_deg[r]));
      }
      if (csr_out_adj_node[i] == s_) {
        dS_ -= csr_out_adj_wgt[i]
          * std::log((float)csr_out_adj_wgt[i]/(csr_out_deg[s_]*csr_in_deg[s_]));
      }
    }
    //dS[r] = dS_;
    if (dS_ < dS[r]) {
      bestS[r] = s_;
      dS[r] = dS_;
    }
  }
}


template <typename Node, typename Weight>
__global__ void calculate_dS_new_out_nodal(float* dS_new_out, Node* s,
                                           unsigned* g_csr_out_adj_ptr,
                                           Node* g_csr_out_adj_node,
                                           unsigned* b_csr_out_adj_ptr,
                                           Node* b_csr_out_adj_node,

					   Weight* b_csr_out_adj_wgt,
                                           Weight* b_csr_out_deg,
                                           
					   
					   
					   Weight* csr_in_deg,
                                           
					   
					   
					   Node* blocks,
					   unsigned N) {

  
  unsigned ni = blockIdx.x * blockDim.x + threadIdx.x;

  if (ni < N) {
   
    unsigned r = blocks[ni];

    unsigned node_e;
    if (ni + 1 < N) {
      node_e = g_csr_out_adj_ptr[ni+1];    	
    }
    else {
      node_e = N;
    }
    Node b;
    for (unsigned i = node_s; i < node_e; i++) {
      b = blocks[g_csr_out_adj_node[i]];
    
    }
  
  }
}


template <typename Node, typename Weight>
__global__ void propose(Node* gpu_random_blocks, Node* gpu_sampling_neighbors1,
                        Node* gpu_sampling_neighbors2, Node* s, 
                        float* gpu_uniform_x, float* gpu_acceptance_prob,
                        Weight* deg, unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    
    if (deg[r] > 0) {
      Node u = gpu_sampling_neighbors1[r];
      if (gpu_uniform_x[r] <= gpu_acceptance_prob[u]) {
        s[r] = gpu_random_blocks[r];
      }
      else {
        Node u = gpu_sampling_neighbors2[r];
        if (deg[u] == 0) {
          s[r] = gpu_random_blocks[r];  // Should be different?
        }
        else {
          s[r] = u;
        }
      }
    }
    else {
      s[r] = gpu_random_blocks[r];
    }
  }

}

template <typename Node, typename Weight>
__global__ void propose_n(Node* gpu_random_blocks, Node* gpu_sampling_neighbors,
                          Node* gpu_sampling_neighbors_nodal, Node* s,
                          float* gpu_uniform_x, float* gpu_acceptance_prob,
                          Weight* node_deg, unsigned N) {

  unsigned ni = blockIdx.x * blockDim.x + threadIdx.x;

  if (ni < N) {
    if (node_deg[ni] > 0) {
      Node u = gpu_sampling_neighbors_nodal[ni];
      if (gpu_uniform_x[ni] <= gpu_acceptance_prob[u]) {
        s[ni] = gpu_random_blocks[ni];
      }
      else {
        s[ni] = gpu_sampling_neighbors[u];
      }
    }
    else {
      s[ni] = gpu_random_blocks[ni];
    }  
  }

}


// ---------------------- Partition -------------------------//
template <typename Node, typename Weight>
void initialize_block_count(Graph<Node, Weight>& g, Block<Node, Weight>& b, unsigned B) {

  std::vector<std::vector<std::pair<Node, Weight>>> Mrow(B);
  std::vector<std::vector<std::pair<Node, Weight>>> Mcol(B);

  b.deg.clear();
  b.deg.resize(B);
  for (unsigned i = 0; i < g.N; i++) {
    Node k1 = g.blocks[i];
    unsigned end;
    if (k1 + 1 < B) {
      end = g.csr_out.adj_ptr[k1+1];
    }
    else {
      end = B;
    }
    for(unsigned j = g.csr_out.adj_ptr[k1]; j < end; j++) {
      Node k2 = g.blocks[g.csr_out.adj_node[j]];
      Mrow[k1].emplace_back(k2, g.csr_out.adj_wgt[j]);
      b.deg[k1] += g.csr_out.adj_wgt[j];
    }
  }

  b.csr_out.adj_ptr.clear();
  b.csr_out.deg.resize(g.N);
  unsigned ptr = 0;
  for (unsigned i = 0; i < g.N; i++) {
    b.csr_out.adj_ptr.emplace_back(ptr);
    for (const auto& [j, w] : Mrow[i]) {
      b.csr_out.adj_node.emplace_back(j);
      b.csr_out.adj_wgt.emplace_back(w);
      b.csr_out.deg[i] += w;
      ptr++;
    }
  }

  for (unsigned i = 0; i < g.N; i++) {
    Node k1 = g.blocks[i];
    unsigned end;
    if (k1 + 1 < B) {
      end = g.csr_in.adj_ptr[k1+1];
    }
    else {
      end = B;
    }
    for(unsigned j = g.csr_in.adj_ptr[k1]; j < end; j++) {
      Node k2 = g.blocks[g.csr_in.adj_node[j]];
      Mcol[k1].emplace_back(k2, g.csr_in.adj_wgt[j]);
      b.deg[k1] += g.csr_in.adj_wgt[j];
    }
  }

  b.csr_in.adj_ptr.clear();
  b.csr_in.deg.resize(g.N);
  ptr = 0;
  for (unsigned i = 0; i < g.N; i++) {
    b.csr_in.adj_ptr.emplace_back(ptr);
    for (const auto& [j, w] : Mcol[i]) {
      b.csr_in.adj_node.emplace_back(j);
      b.csr_in.adj_wgt.emplace_back(w);
      b.csr_in.deg[i] += w;
      ptr++;
    }
  }
  

}



template <typename Node, typename Weight>
void carry_out_best_merge(Graph<Node, Weight>& g, 
		          std::vector<float>& dS, 
			  std::vector<Node>& S, 
			  unsigned B, unsigned bToMerges) {

  std::vector<Node> bestMerges(B);
  std::vector<int> blockMap(B);
  std::vector<Node> remainBlocks;
  std::set<Node> seen;

  std::iota(bestMerges.begin(), bestMerges.end(), 0);
  std::iota(blockMap.begin(), blockMap.end(), 0);
  std::sort(bestMerges.begin(), bestMerges.end(), [&] (unsigned i, unsigned j) {
    return dS[i] < dS[j];
  });

  unsigned numMerges = 0;
  unsigned counter = 0;
  while (numMerges < bToMerges) {
    Node mergeFrom = bestMerges[counter];
    Node mergeTo = blockMap[S[mergeFrom]];
    counter++;
    if (mergeTo != mergeFrom) {
      for (unsigned i = 0; i < B; i++) {
      	if (blockMap[i] == mergeFrom) {
	  blockMap[i] = mergeTo;
	}
      }
      for (unsigned i = 0; i < B; i++) {
      	if (g.blocks[i] == mergeFrom) {
	  g.blocks[i] = mergeTo;
	}
      }
      numMerges++;
    }
  }

  for (const auto& b : g.blocks) {
    if (seen.find(b) == seen.end()) {
      seen.insert(b);
    }
  }

  remainBlocks.insert(remainBlocks.end(), seen.begin(), seen.end());
  std::sort(remainBlocks.begin(), remainBlocks.end());
  blockMap.clear();
  blockMap.resize(B, -1);
  for (unsigned i = 0; i < remainBlocks.size(); i++) {
    blockMap[remainBlocks[i]] = i;
  }
  for (auto& b : g.blocks) {
    b = blockMap[b];
  }

}


template <typename Node, typename Weight>
void propose_block_merge(Block<Node, Weight>& b, unsigned B, 
		         std::vector<Node>& S,
	                 std::vector<float>& dS,
			 unsigned numProposals) {
  
  unsigned block_size = 256;
  unsigned num_blocks = (B + block_size - 1) / block_size;
 
  // Create stream
  cudaStream_t s1, s2, s3, s4, s5;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaStreamCreate(&s3);
  cudaStreamCreate(&s4);
  cudaStreamCreate(&s5);

  // device data
  unsigned* gpu_csr_out_adj_ptr;
  Node* gpu_csr_out_adj_node;
  Weight* gpu_csr_out_adj_wgt;
  Weight* gpu_csr_out_deg;

  unsigned* gpu_csr_in_adj_ptr;
  Node* gpu_csr_in_adj_node;
  Weight* gpu_csr_in_adj_wgt;
  Weight* gpu_csr_in_deg;

  Weight* gpu_deg;
  Node* gpu_random_blocks;
  Node* gpu_sampling_neighbors1;
  Node* gpu_sampling_neighbors2;
  float* gpu_uniform_x;
  float* gpu_acceptance_prob;
  Node* proposed_blocks;
  Node* best_proposed_blocks;
  float* gpu_dS_out;
  float* gpu_dS_in;
  float* gpu_dS_new_out;
  float* gpu_dS_new_in;
  float* gpu_dS;


  // Allocate GPU memory
  cudaMallocAsync(&gpu_csr_out_adj_ptr, sizeof(unsigned)*b.csr_out.adj_ptr.size(), s1);  
  cudaMallocAsync(&gpu_csr_out_adj_node, sizeof(Node)*b.csr_out.adj_node.size(), s1);
  cudaMallocAsync(&gpu_csr_out_adj_wgt, sizeof(Weight)*b.csr_out.adj_wgt.size(), s1);
  cudaMallocAsync(&gpu_csr_out_deg, sizeof(Weight)*b.csr_out.deg.size(), s1);
  cudaMallocAsync(&gpu_csr_in_adj_ptr, sizeof(unsigned)*b.csr_in.adj_ptr.size(), s1);
  cudaMallocAsync(&gpu_csr_in_adj_node, sizeof(Node)*b.csr_in.adj_node.size(), s1);
  cudaMallocAsync(&gpu_csr_in_adj_wgt, sizeof(Weight)*b.csr_in.adj_wgt.size(), s1);
  cudaMallocAsync(&gpu_csr_in_deg, sizeof(Weight)*b.csr_in.deg.size(), s1);
  cudaMallocAsync(&gpu_deg, sizeof(Weight)*B, s1);
  cudaMallocAsync(&gpu_random_blocks, sizeof(Node)*B, s1);
  cudaMallocAsync(&gpu_sampling_neighbors1, sizeof(Node)*B, s1);
  cudaMallocAsync(&gpu_sampling_neighbors2, sizeof(Node)*B, s1);
  cudaMallocAsync(&gpu_uniform_x, sizeof(float)*B, s1);
  cudaMallocAsync(&gpu_acceptance_prob, sizeof(float)*B, s1);
  cudaMallocAsync(&proposed_blocks, sizeof(Node)*B, s1);
  cudaMallocAsync(&best_proposed_blocks, sizeof(Node)*B, s1);
  cudaMallocAsync(&gpu_dS_out, sizeof(float)*B, s1);
  cudaMallocAsync(&gpu_dS_in, sizeof(float)*B, s1);
  cudaMallocAsync(&gpu_dS_new_out, sizeof(float)*B, s1);
  cudaMallocAsync(&gpu_dS_new_in, sizeof(float)*B, s1);
  cudaMallocAsync(&gpu_dS, sizeof(float)*B, s1);


  // transfer data
  cudaMemcpyAsync(gpu_csr_out_adj_ptr, b.csr_out.adj_ptr.data(), sizeof(unsigned)*b.csr_out.adj_ptr.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_out_adj_node, b.csr_out.adj_node.data(), sizeof(Node)*b.csr_out.adj_node.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_out_adj_wgt, b.csr_out.adj_wgt.data(), sizeof(Weight)*b.csr_out.adj_wgt.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_out_deg, b.csr_out.deg.data(), sizeof(Weight)*b.csr_out.deg.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_in_adj_ptr, b.csr_in.adj_ptr.data(), sizeof(unsigned)*b.csr_in.adj_ptr.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_in_adj_node, b.csr_in.adj_node.data(), sizeof(Node)*b.csr_in.adj_node.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_in_adj_wgt, b.csr_in.adj_wgt.data(), sizeof(Weight)*b.csr_in.adj_wgt.size(), cudaMemcpyDefault, s1);
  cudaMemcpyAsync(gpu_csr_in_deg, b.csr_in.deg.data(), sizeof(Weight)*b.csr_in.deg.size(), cudaMemcpyDefault, s1);


  for(unsigned _ = 0; _ < numProposals; _++) {

    random_block_generator<Node, Weight> <<<num_blocks, block_size, 0, s1>>>(
      gpu_random_blocks, B
    );

    uniform_number_generator<Node, Weight> <<<num_blocks, block_size, 0, s2>>>(
      gpu_uniform_x, B
    );

    sample_neighbors<Node, Weight> <<<num_blocks, block_size, 0, s3>>>(
      gpu_sampling_neighbors1, gpu_csr_out_adj_ptr, gpu_csr_out_adj_node, gpu_csr_out_adj_wgt, 
      gpu_csr_in_adj_ptr, gpu_csr_in_adj_node, gpu_csr_in_adj_wgt, gpu_deg, B
    );

    sample_neighbors<Node, Weight> <<<num_blocks, block_size, 0, s4>>>(
      gpu_sampling_neighbors2, gpu_csr_out_adj_ptr, gpu_csr_out_adj_node, gpu_csr_out_adj_wgt,
      gpu_csr_in_adj_ptr, gpu_csr_in_adj_node, gpu_csr_in_adj_wgt, gpu_deg, B
    );

    calculate_acceptance_prob<Node, Weight> <<<num_blocks, block_size, 0, s5>>>(
      gpu_acceptance_prob, gpu_deg, B
    );

    cudaDeviceSynchronize();

    propose<Node, Weight> <<<num_blocks, block_size, 0, s1>>>(
      gpu_random_blocks, gpu_sampling_neighbors1, gpu_sampling_neighbors2,
      proposed_blocks, gpu_uniform_x, gpu_acceptance_prob, gpu_deg, B
    );

    cudaDeviceSynchronize();

    calculate_dS_out<Node, Weight> <<<num_blocks, block_size, 0, s1>>>(
      gpu_dS_out, gpu_csr_out_adj_ptr, gpu_csr_out_adj_node, gpu_csr_out_adj_wgt,
      gpu_csr_out_deg, gpu_csr_in_deg, B
    );

    calculate_dS_in<Node, Weight> <<<num_blocks, block_size, 0, s2>>>(
      gpu_dS_in, gpu_csr_in_adj_ptr, gpu_csr_in_adj_node, gpu_csr_in_adj_wgt,
      gpu_csr_in_deg, gpu_csr_out_deg, B
    );

    calculate_dS_new_out<Node, Weight> <<<num_blocks, block_size, 0, s3>>>(
      gpu_dS_new_out, proposed_blocks, gpu_csr_out_adj_ptr, gpu_csr_out_adj_node, 
      gpu_csr_out_adj_wgt, gpu_csr_out_deg, gpu_csr_in_deg, B
    );

    calculate_dS_new_in<Node, Weight> <<<num_blocks, block_size, 0, s4>>>(
      gpu_dS_new_in, proposed_blocks, gpu_csr_in_adj_ptr, gpu_csr_in_adj_node,
      gpu_csr_in_adj_wgt, gpu_csr_in_deg, gpu_csr_out_deg, B
    );

    cudaDeviceSynchronize();

    calculate_dS_overall<Node, Weight> <<<num_blocks, block_size, 0, s4>>>(
      gpu_dS, gpu_dS_out, gpu_dS_in, gpu_dS_new_out, gpu_dS_new_in,
      gpu_csr_out_adj_ptr, gpu_csr_out_adj_node, gpu_csr_out_adj_wgt, gpu_csr_out_deg, 
      gpu_csr_in_deg, proposed_blocks, best_proposed_blocks, B
    );

    cudaDeviceSynchronize();
  }

  // get the result back
  cudaMemcpyAsync(&dS[0], gpu_dS, sizeof(float)*B, cudaMemcpyDefault, s1);
  cudaMemcpyAsync(&S[0], best_proposed_blocks, sizeof(Node)*B, cudaMemcpyDefault, s1);


  cudaFreeAsync(gpu_csr_out_adj_ptr, s1);
  cudaFreeAsync(gpu_csr_out_adj_node, s1);
  cudaFreeAsync(gpu_csr_out_adj_wgt, s1);
  cudaFreeAsync(gpu_csr_out_deg, s1);
  cudaFreeAsync(gpu_csr_in_adj_ptr, s1);
  cudaFreeAsync(gpu_csr_in_adj_node, s1);
  cudaFreeAsync(gpu_csr_in_adj_wgt, s1);
  cudaFreeAsync(gpu_csr_in_deg, s1);
  cudaFreeAsync(gpu_deg, s1);
  cudaFreeAsync(gpu_random_blocks, s1);
  cudaFreeAsync(gpu_sampling_neighbors1, s1);
  cudaFreeAsync(gpu_sampling_neighbors2, s1);
  cudaFreeAsync(gpu_uniform_x, s1);
  cudaFreeAsync(gpu_acceptance_prob, s1);
  cudaFreeAsync(proposed_blocks, s1);
  cudaFreeAsync(best_proposed_blocks, s1);
  cudaFreeAsync(gpu_dS_out, s1);
  cudaFreeAsync(gpu_dS_in, s1);
  cudaFreeAsync(gpu_dS_new_out, s1);
  cudaFreeAsync(gpu_dS_new_in, s1);
  cudaFreeAsync(gpu_dS, s1);
  
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
  cudaStreamDestroy(s4);
  cudaStreamDestroy(s5);

}


template <typename Node, typename Weight>
void propose_nodal_move() {


  random_block_generator_nodal<Node, Weight> <<<num_blocks, block_size, 0, s1>>>(
    gpu_random_blocks, B, g.N
  );

  uniform_number_generator<Node, Weight> <<<num_blocks, block_size, 0, s2>>>(
    gpu_uniform_x, g.N
  );

  sample_neighbors<Node, Weight> <<<num_blocks, block_size, 0, s3>>>(
    gpu_sampling_neighbors, gpu_b_csr_out_adj_ptr, gpu_b_csr_out_adj_node, gpu_b_csr_out_adj_wgt,
    gpu_b_csr_in_adj_ptr, gpu_b_csr_in_adj_node, gpu_b_csr_in_adj_wgt, gpu_b_deg, B
  );

  sample_neighbors_nodal<Node, Weight> <<<num_blocks, block_size, 0, s4>>>(
    gpu_sampling_neighbors_nodal, gpu_g_csr_out_adj_ptr, gpu_g_csr_out_adj_node,
    gpu_g_csr_out_adj_wgt, gpu_g_csr_in_adj_ptr, gpu_g_csr_in_adj_node,
    gpu_g_csr_in_adj_wgt, gpu_g_deg, gpu_blocks, N
  );

  calculate_acceptance_prob<Node, Weight> <<<num_blocks, block_size, 0, s5>>>(
    gpu_acceptance_prob, gpu_deg, B
  );

  cudaDeviceSynchronize();

  /////////////////////////////
  ////////////////////////////
  // add BS


    propose<Node, Weight> <<<num_blocks, block_size, 0, s1>>>(
      gpu_random_blocks, gpu_sampling_neighbors1, gpu_sampling_neighbors2,
      proposed_blocks, gpu_uniform_x, gpu_acceptance_prob, gpu_deg, B
    );

    cudaDeviceSynchronize();

  ///////////////////////////
    ///////////////


  calculate_dS_out(float* dS_out,
                                 unsigned* csr_out_adj_ptr,
                                 Node* csr_out_adj_node,
                                 Weight* csr_out_adj_wgt,
                                 Weight* csr_out_deg,
                                 Weight* csr_in_deg,
                                 unsigned B)

  calculate_dS_in(float* dS_in,
                                unsigned* csr_in_adj_ptr,
                                Node* csr_in_adj_node,
                                Weight* csr_in_adj_wgt,
                                Weight* csr_in_deg,
                                Weight* csr_out_deg,
                                unsigned B)

}

int main (int argc, char *argv[]) {
 

  unsigned numProposals = 10;
  float blockReduction = 0.5;


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
    default:
      std::cerr << "usage: ./run [Number of Nodes=1000/5000/20000/50000]\n";
      std::exit(1);
  }

  Graph g = load_graph_from_tsv<unsigned, long>(FileName);
  g.blocks.resize(g.N);
  std::iota(g.blocks.begin(), g.blocks.end(), 0);
  std::cout << "Number of nodes: " << g.N << std::endl;
  std::cout << "Number of edges: " << g.E << std::endl;

  Block<unsigned, long> b;
  unsigned B = g.N;
  unsigned bToMerges = B * blockReduction;

  initialize_block_count(g, b, B);

  std::vector<unsigned> S(B);
  std::vector<float> dS(B);
  propose_block_merge(b, B, S, dS, numProposals);


  carry_out_best_merge(g, dS, S, B, bToMerges);
  
  B -= bToMerges;



  return 0;
} 
