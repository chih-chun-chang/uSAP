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
  Graph() : N(0), E(0) {}
};

template <typename Node, typename Weight>
bool compare_s(const Edge<Node, Weight>& e1, const Edge<Node, Weight>&& e2) {
  return e1.s < e2.s;
}

template <typename Node, typename Weight>
bool compare_t(const Edge<Node, Weight>&& e1, const Edge<Node, Weight>&& e2) {
  return e1.t < e2.t;
}

template <typename Node, typename Weight>
void load_graph_from_tsv(const std::string& FileName, Csr<Node, Weight>& csr) {
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
  
  std::sort(g.edges.begin(), g.edges.end(), compare_s);
  g.csr_out.adj_ptr.emplace_back(0);
  g.csr_out.deg.resize(g.N);
  s = 0;
  for (unsigned i = 0; i < g.E; i++) {
    if (g.edges[i].s - 1 != s) {
      s++;
      g.csr_out.adj_ptr.emplace_back(i);
    }
    g.csr_out.adj_node.emplace_back(g.edges[i].t - 1);
    g.csr_out.adj_wgt.emplace_back(g.edges[i].w);
    g.csr_out.deg[g.edges[i].s - 1] += g.edges[i].w;
    g.deg[g.edges[i].s - 1] += g.edges[i].w;
  }

  std::sort(g.edges.begin(), g.edges.end(), compare_t);
  g.csr_in.adj_ptr.emplace_back(0);
  g.csr_in.deg.resize(g.N);
  t = 0;
  for (unsigned i = 0; i < g.E; i++) {
    if (g.edges[i].t - 1 != t) {
      t++;
      g.csr_in.adj_ptr.emplace_back(i);
    }
    g.csr_in.adj_node.emplace_back(g.edges[i].s - 1);
    g.csr_in.adj_wgt.emplace_back(g.edges[i].w);
    g.csr_in.deg[g.edges[i].t - 1] += g.edges[i].w;
    g.deg[g.edges[i].t - 1] += g.edges[i].w;
  }
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
                                 Csr<Node, Weight>* csr_out,
                                 Csr<Node, Weight>* csr_in, 
                                 Weight* deg, unsigned B) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    float random = curand_uniform(&state);

    unsigned out_s, out_e;
    unsigned in_s, in_e;
    out_s = csr_out[idx].adj_ptr;
    in_s = csr_in[idx].adj_ptr;
    if (idx + 1 < B) {
      out_e = csr_out[idx+1].adj_ptr;
      in_e = csr_in[idx+1].adj_ptr;
    }
    else {
      out_e = B;
      in_e = B; 
    }

    float prob_sum = 0.0f;
    unsigned neighbor;
    unsigned find = 0;
    for (unsigned i = out_s; i < out_e; i++) {
      prob_sum += (float)csr_out[i].adj_wgt[i]/deg[idx];
      if (random > prob_sum) {
        neighbor = csr_out[i].adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (unsigned i = in_s; i < in_e; i++) {
        prob_sum += (float)csr_in[i].adj_wgt/deg[idx];
        if (random > prob_sum) {
          neighbor = csr_in[i].adj_node;
          break;
        }
      }
    }

    gpu_sampling_neighbors[idx] = neighbor;
     
  }

}


template <typename Node, typename Weight>
__global__ void calculate_dS_out(float* dS_out, 
                                 Csr<Node, Weight>* csr_out, 
                                 Csr<Node, Weight>* csr_in, 
                                 unsigned B) {
  
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    unsigned out_s = csr_out[idx].adj_ptr;
    unsigned out_e;
    if (idx + 1 < B) {
      out_e = csr_out[idx+1].adj_ptr;
    }
    else {
      out_e = B;
    }
    float dS = 0;
    for (unsigned i = out_s; i < out_e; i++) {
      dS += (float)csr_out[i].adj_wgt * std::log((float)csr_out[i].adj_wgt
          / (csr_out[idx].deg * csr_in[csr_out[i].adj_node].deg));
    }
    dS_out[idx] = dS;
  }

}

template <typename Node, typename Weight>
__global__ void calculate_dS_in(float* dS_in, 
                                Csr<Node, Weight>* csr_out, 
                                Csr<Node, Weight>* csr_in,
                                unsigned B) {
    
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    unsigned in_s = csr_in[idx].adj_ptr;
    unsigned in_e;
    if (idx + 1 < B) {
      in_e = csr_in[idx+1].adj_ptr;
    }
    else {
      in_e = B;
    }
    float dS = 0;
    for (unsigned i = in_s; i < in_e; i++) {
      dS += (float)csr_in[i].adj_wgt * std::log((float)csr_in[i].adj_wgt
          / (csr_out[csr_in[i].adj_node].deg * csr_in[idx].deg));
    }
    dS_in[idx] = dS;
  }

}

template <typename Node, typename Weight>
__global__ void calculate_dS_new_out(float* dS_new_out, float* dS_out, Node* s,
                                     Csr<Node, Weight>* csr_out, 
                                     Csr<Node, Weight>* csr_in, unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    unsigned s_ = s[r]; 
    unsigned out_r_s, out_r_e;
    unsigned out_s_s, out_s_e;
    out_r_s = csr_out[r].adj_ptr;
    out_s_s = csr_out[s_].adj_ptr;
    if (r + 1 < B) {
      out_r_e = csr_out[r+1].adj_ptr;
    }   
    else {
      out_r_e = B;
    }    
    if (s_ + 1 < B) {
      out_s_e = csr_out[s_+1].adj_ptr;
    }
    else {
      out_s_e = B;
    }
    unsigned i = out_r_s, j = out_s_s;
    float dS = 0;
    Weight w;
    Weight dout = csr_out[r].deg + csr_out[s_].deg;
    Node n;
    while (i < out_r_e && j < out_s_e) {
      if (csr_out[i].adj_node < csr_out[j].adj_node) {
        w = csr_out[i].adj_wgt;
        n = csr_out[i].adj_node;
        i++;
      }
      else if (csr_out[i].adj_node > csr_out[j].adj_node) {
        w = csr_out[j].adj_wgt;
        n = csr_out[j].adj_node;
        j++;
      }
      else {
        w = csr_out[i].adj_wgt + csr_out[j].adj_wgt;
        n = csr_out[i].adj_node;
        i++;
        j++;
      }
      dS -= w * std::log((float)w/(dout*csr_in[n].deg));
    }
    for (; i < out_r_e; i++) {
      w = csr_out[i].adj_wgt;
      n = csr_out[i].adj_node;
      dS -= w * std::log((float)w/(dout*csr_in[n].deg));
    }
    for (; j < out_s_e; j++) {
      w = csr_out[j].adj_wgt;
      n = csr_out[j].adj_node;
      dS -= w * std::log((float)w/(dout*csr_out[n].deg));
    }
    dS_new_out[r] = dS;
  }

}


template <typename Node, typename Weight>
__global__ void calculate_dS_new_in(float* dS_new_in, float* dS_in, Node* s,
                                    Csr<Node, Weight>* csr_out, 
                                    Csr<Node, Weight>* csr_in, unsigned B) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    unsigned s_ = s[r];
    unsigned in_r_s, in_r_e;
    unsigned in_s_s, in_s_e;
    in_r_s = csr_in[r].adj_ptr;
    in_s_s = csr_in[s_].adj_ptr;
    if (r + 1 < B) {
      in_r_e = csr_in[r+1].adj_ptr;
    } 
    else {
      in_r_e = B;
    } 
    if (s_ + 1 < B) {
      in_s_e = csr_in[s_+1].adj_ptr;
    }
    else {
      in_s_e = B;
    }
    unsigned i = in_r_s, j = in_s_s;
    float dS = 0;
    Weight w;
    Weight din = csr_in[r].deg + csr_in[s_].deg;
    Node n;
    while (i < in_r_e && j < in_s_e) {
      if (csr_in[i].adj_node < csr_in[j].adj_node) {
        w = csr_in[i].adj_wgt;
        n = csr_in[i].adj_node;
        i++;
      }
      else if (csr_in[i].adj_node > csr_in[j].adj_node) {
        w = csr_in[j].adj_wgt;
        n = csr_in[j].adj_node;
        j++;
      }
      else {
        w = csr_in[i].adj_wgt + csr_in[j].adj_wgt;
        n = csr_in[i].adj_node;
        i++;
        j++;
      }
      dS -= w * std::log((float)w/(csr_out[n].deg*din));
    }
    for (; i < in_r_e; i++) {
      w = csr_in[i].adj_wgt;
      n = csr_in[i].adj_node;
      dS -= w * std::log((float)w/(csr_out[n].deg*din));
    }
    for (; j < in_s_e; j++) {
      w = csr_in[j].adj_wgt;
      n = csr_in[j].adj_node;
      dS -= w * std::log((float)w/(csr_out[n].deg*din));
    }
    dS_new_in[r] = dS;
  }

}

template <typename Node, typename Weight>
__global__ void calculate_dS_overall(float* dS, float* dS_out, float* dS_in,
                                     float* dS_new_out, float* dS_new_in, 
                                     Csr<Node, Weight>* csr_out, 
                                     Csr<Node, Weight>* csr_in, Node* s,
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
    
    unsigned out_r_s, out_r_e;
    unsigned out_s_s, out_s_e;
    out_r_s = csr_out[r].adj_ptr;
    out_s_s = csr_out[s_].adj_ptr;
    if (r + 1 < B) {
      out_r_e = csr_out[r+1].adj_ptr;
    } 
    else {
      out_r_e = B;
    }
    if (s_ + 1 < B) {
      out_s_e = csr_out[s_+1].adj_ptr;
    }
    else {
      out_s_e = B;
    }
    for (unsigned i = 0; i < out_r_e; i++) {
      if (csr_out[i].adj_node == r) {
        dS_ -= csr_out[i].adj_wgt
          * std::log((float)csr_out[i].adj_wgt/(csr_out[r].deg*csr_in[r].deg));
      }
      if (csr_out[i].adj_node == s_) {
        dS_ -= csr_out[i].adj_wgt
          * std::log((float)csr_out[i].adj_wgt/(csr_out[r].deg*csr_in[s_].deg));
      }
    }
    for (unsigned i = 0; i < out_s_e; i++) {
      if (csr_out[i].adj_node == r) {
        dS_ -= csr_out[i].adj_wgt
          * std::log((float)csr_out[i].adj_wgt/(csr_out[s_].deg*csr_in[r].deg));
      }
      if (csr_out[i].adj_node == s_) {
        dS_ -= csr_out[i].adj_wgt
          * std::log((float)csr_out[i].adj_wgt/(csr_out[s_].deg*csr_in[s_].deg));
      }
    }
    dS[r] = dS_;
  }
}






// ---------------------- Partition -------------------------//
/*
template <typename Node, typename Weight>
void propose_block_merge(
  Weight* M, // inter-block edge count (B*B)
  Node* G,
  unsigned B // current block number
) {

  cudaStream_t stream;
  cudaStreamCreate(&stream); // TODO: create multiple stream for num_proposal??

  Weight* gpu_M;
  Weight* gpu_Mt;
  Weight* dout; // degree out for each block (B*1)
  Weight* din;  // degree in for each block (B*1)
  Weight* d;
  float* P; // probability matrix (B*B)
  float* x;

  Node* S; // proposal for each block (B*1)
  unsigned* n; // random index (B*1)
  Node* dG;
  Node* u; 

  // TODO: malloc only once
  cudaMallocAsync(&gpu_M, sizeof(Weight)*B*B, stream);
  cudaMallocAsync(&gpu_Mt, sizeof(Weight)*B*B, stream);
  cudaMallocAsync(&dout, sizeof(Weight)*B, stream);
  cudaMallocAsync(&din, sizeof(Weight)*B, stream);
  cudaMallocAsync(&d, sizeof(Weight)*B, stream);
  cudaMallocAsync(&P, sizeof(float)*B*B, stream);
  cudaMallocAsync(&S, sizeof(Node)*B, stream);
  cudaMallocAsync(&n, sizeof(unsigned)*B, stream);
  cudaMallocAsync(&dG, sizeof(Node)*B, stream);
  cudaMallocAsync(&u, sizeof(Node)*B, stream);
  cudaMallocAsync(&x, sizeof(float)*B, stream);

  cudaMemcpyAsync(gpu_M, M, sizeof(Weight)*B*B, cudaMemcpyDefault, stream);
  cudaMemcpyAsync(dG, G, sizeof(Node)*B, cudaMemcpyDefault, stream);

  dim3 dimGrid(B/TILE_DIM, B/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  // dout = MI
  matrix_row_reduce<<<dimGrid, dimBlock, 0, stream>>>(
    dout, gpu_M, B
  );
  
  // din = MtI (TODO: matrix_col_reduce)
  matrix_transpose<<<dimGrid, dimBlock, 0, stream>>>(
    din, gpu_M, B
  );
  matrix_row_reduce<<<dimGrid, dimBlock, 0, stream>>>(
    din, din, B
  );
  
  // d = din + dout
  vector_addition<<<dimGrid, dimBlock, 0, stream>>>(
	d, din, dout, B
  );
  
  // P = (M+Mt)/d
  matrix_transpose<<<dimGrid, dimBlock, 0, stream>>>(
    gpu_Mt, gpu_M, B
  );
  matrix_addition<<<dimGrid, dimBlock, 0, stream>>>(
    gpu_Mt, gpu_Mt, gpu_M, B
  );
  matrix_row_divide<<<dimGrid, dimBlock, 0, stream>>>(
	P, gpu_Mt, d, B
  );

  // if d[i] == 0 -> S[i] = rand(B)
  vector_random_number_generator<<<1, B, 0, stream>>>(
    S, d, B
  );

  // if d[i] != 0 -> ...
  // 1. multinomial based on P
  matrix_discrete_distribution<<<1, B, 0, stream>>>(
    // TODO:...
  ); 
  
  // 2. u = G[n]
  vector_map<<<1, B, 0, stream>>>(
    u, dG, n, B
  );

  // 3. x ~ U(0,1)
  vector_uniform_number_generator<<<1, B, 0, stream>>>(
    x, B
  );

  // 4. prob = B/(u+B)
  calculate_prob<<<1, B, 0, stream>>>(
    prob, d, B
  );
      
  // 5. x <= prob[u] -> S[i] = rand(B)

  // 6. x > prob[u] ->
  //      rowsum(prob) == prob[i] -> S[i] = rand(B)
  //      prob / rowsum(prob) -> S[i] = multinomial

}
*/

int main (int argc, char *argv[]) {
  
  //std::string FileName("./Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar");
 
  //if(argc != 2) {
  //  std::cerr << "usage: ./run [Number of Nodes]\n";
  //  std::exit(1);
  //}

  //int num_nodes = std::stoi(argv[1]);

  //switch(num_nodes)  {
  //  case 1000:
  //    FileName += "_1000_nodes";
  //    break;
  //  case 5000:
  //    FileName += "_5000_nodes";
  //    break;
  //  default:
  //    std::cerr << "usage: ./run [Number of Nodes=1000/5000/20000/50000]\n";
  //    std::exit(1);
  //}

  //Graph g = load_graph_from_tsv(FileName);
  //std::cout << "Number of nodes: " << g.N << std::endl;
  //std::cout << "Number of edges: " << g.E << std::endl;

  //std::srand(std::time(nullptr));

  //unsigned width = 1024;
  //
  //dim3 dimGrid(width/TILE_DIM, width/TILE_DIM, 1);
  //dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  //std::vector<int> M(width*width);
  //for (unsigned i = 0; i < width; i++) {
  //  for (unsigned j = 0; j < width; j++) {
  //    M[i*width + j] = std::rand() % 100;
  //  }
  //}


  ////int* cpu_out = new int[width*width];
  ////matrix_transpose_seq(cpu_out, M.data(), width);
  ////matrix_addition_seq(cpu_out, cpu_out, M.data(), width);
  ////int* cpu_out = new int[width];
  ////float* cpu_out2 = new float[width*width];
  ////matrix_row_reduce_seq(cpu_out, M.data(), width);
  ////matrix_row_divide_seq(cpu_out2, M.data(), cpu_out, width);

  ////////////////
  //cudaStream_t stream;
  //cudaStreamCreate(&stream);

  //int* gpu_in;
  //int* gpu_out;
  //float* gpu_out2;
  //float* res = new float[width*width];
  ////int* res = new int[width];

  //cudaMallocAsync(&gpu_in, sizeof(int)*width*width, stream);
  //cudaMallocAsync(&gpu_out, sizeof(int)*width, stream);
  //cudaMallocAsync(&gpu_out2, sizeof(int)*width*width, stream);
  //cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width*width, cudaMemcpyDefault, stream);


  ////matrix_transposeDiagonal<<<dimGrid, dimBlock, 0, stream>>>(
  ////  gpu_out, gpu_in, width
  ////);
  ////matrix_addition<<<dimGrid, dimBlock, 0, stream>>>(
  ////  gpu_out, gpu_out, gpu_in, width
  ////);
  //matrix_row_reduce<<<dimGrid, dimBlock, 0, stream>>>(
  //  gpu_out, gpu_in, width
  //);
  //matrix_row_divide<<<dimGrid, dimBlock, 0, stream>>>(
	//gpu_out2, gpu_in, gpu_out, width
  //);

  //cudaMemcpyAsync(res, gpu_out2, sizeof(int)*width*width, cudaMemcpyDefault, stream);

  //cudaStreamSynchronize(stream);

  //////////////


  /////////////
  //cudaFreeAsync(gpu_in, stream);
  //cudaFreeAsync(gpu_out, stream);
  //cudaFreeAsync(gpu_out2, stream);
  //cudaStreamDestroy(stream);

  ////delete[] cpu_out; 
  //delete[] res; 


  return 0;
} 
