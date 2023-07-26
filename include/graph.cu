#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cassert>
#include <cstdlib>
#include <ctime>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

template <typename Node, typename Weight>
struct Edge {
  Node   s;
  Node   t;
  Weight w;
  Edge(Node s, Node t, Weight w) : s(s), t(t), w(w) {}
};

template <typename Node, typename Weight>
struct Graph {
  // number of node and edge
  long N;
  long E;
  std::vector<Edge<Node, Weight>> edges;
  // adjacency matrix
  std::vector<Weight> A;
  Graph() : N(0), E(0) {}
};

template <typename Node, typename Weight>
Graph<Node, Weight> load_graph_from_tsv(const std::string& FileName) {
  std::ifstream file(FileName + ".tsv");
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  Graph<Node, Weight> g;
  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;
  Node s, t;
  Weight w;
  while (std::getline(file, line)) {
    int start = 0;
    int tab_pos = line.find('\t');
    s = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    t = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    w = static_cast<Weight>(std::stof(line.substr(start, tab_pos - start)));
    g.edges.emplace_back(s, t, w);
    if (s > g.N) g.N = s;
  }
  file.close();

  g.E = g.edges.size();
  g.A.resize(g.N*g.N);
  for (const auto& e : g.edges) {
    g.A[e.s*g.N + e.t] += e.w;
    g.A[e.t*g.N + e.s] += e.w;
  }
  return g;
}// end of load_graph_from_tsv

// --------------------- sequential code -----------------//

void matrix_transpose_seq(int* odata, int* idata, unsigned width) {

  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < width; j++) {
      odata[i + j*width] = idata[i*width + j];
    }
  }
}

void matrix_addition_seq(int* odata, int* idata1, int* idata2, unsigned width) {

  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < width; j++) {
      odata[i + j*width] = idata1[i + j*width] + idata2[i + j*width];
    }
  }
}

void matrix_row_reduce_seq(int* odata, int* idata, unsigned width) {

  for (unsigned i = 0; i < width; i++) {
    odata[i] = 0;
    for (unsigned j = 0; j < width; j++) {
      odata[i] += idata[i + j*width];
    }
  }

}

void check(int* odata, int* idata, unsigned width) {
  for (unsigned i = 0; i < width; i++) { 
    for (unsigned j = 0; j < width; j++) {
      if (odata[i*width + j] != idata[i*width + j]) {
        std::cout << odata[i*width + j] << ", " << idata[i*width + j] << "\n";
        assert(odata[i*width + j] == idata[i*width + j]);
      }
    }
  }
} 

// ----------------------CUDA kernel ---------------------//


__global__ void matrix_transposeDiagonal(int* odata, int* idata, unsigned width) {

  __shared__ int tile[TILE_DIM][TILE_DIM+1];

  unsigned blockIdx_x, blockIdx_y;

  // reordering
  blockIdx_y = blockIdx.x;
  blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;

  unsigned x = threadIdx.x + blockIdx_x * TILE_DIM;
  unsigned y = threadIdx.y + blockIdx_y * TILE_DIM;
  unsigned index_in = x + y*width;

  x = threadIdx.x + blockIdx_y*TILE_DIM;
  y = threadIdx.y + blockIdx_x*TILE_DIM;
  unsigned index_out = x + y*width;

  for (unsigned i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i*width];
  }

  __syncthreads();

  for (unsigned i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
    odata[index_out + i*width] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void matrix_addition(int* odata, int* idata1, int* idata2, unsigned width) {

  __shared__ int tile[TILE_DIM][TILE_DIM+1];

  unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
  unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;
  unsigned index = x + y*width;

  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.x][threadIdx.y + i] = idata1[index + i*width] + idata2[index + i*width];
  }

  __syncthreads();

  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
    odata[index + i*width] = tile[threadIdx.x][threadIdx.y + i];
  }
}


__global__ void matrix_row_reduce(int* odata, int* idata, unsigned width) {

  __shared__ int tile[TILE_DIM][TILE_DIM+1];

  unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
  unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;
  unsigned index = x + y*width;

  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.x][threadIdx.y + i] = idata[index + i*width];
  }

  __syncthreads();

  //TODO:

}


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

  std::srand(std::time(nullptr));

  unsigned width = 1024;
  
  dim3 dimGrid(width/TILE_DIM, width/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  std::vector<int> M(width*width);
  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < width; j++) {
      M[i*width + j] = std::rand() % 100;
    }
  }

  //int* cpu_out = new int[width*width];
  //matrix_transpose_seq(cpu_out, M.data(), width);
  //matrix_addition_seq(cpu_out, cpu_out, M.data(), width);
  int* cpu_out = new int[width];
  matrix_row_reduce_seq(cpu_out, M.data(), width);


  //////////////
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int* gpu_in;
  int* gpu_out;
  //int* res = new int[width*width];
  int* res = new int[width];

  cudaMallocAsync(&gpu_in, sizeof(int)*width*width, stream);
  //cudaMallocAsync(&gpu_out, sizeof(int)*width*width, stream);
  cudaMallocAsync(&gpu_out, sizeof(int)*width, stream);

  //cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width*width, cudaMemcpyDefault, stream);
  cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width, cudaMemcpyDefault, stream);

  //matrix_transposeDiagonal<<<dimGrid, dimBlock, 0, stream>>>(
  //  gpu_out, gpu_in, width
  //);
  //matrix_addition<<<dimGrid, dimBlock, 0, stream>>>(
  //  gpu_out, gpu_out, gpu_in, width
  //);
  matrix_row_reduce<<<dimGrid, dimBlock, 0, stream>>>(
    gpu_out, gpu_in, width
  );

  //cudaMemcpyAsync(res, gpu_out, sizeof(int)*width*width, cudaMemcpyDefault, stream);
  cudaMemcpyAsync(res, gpu_out, sizeof(int)*width, cudaMemcpyDefault, stream);

  cudaStreamSynchronize(stream);

  ////////////

  check(cpu_out, res, width);

  ///////////
  cudaFreeAsync(gpu_in, stream);
  cudaFreeAsync(gpu_out, stream);
  cudaStreamDestroy(stream);

  delete[] cpu_out; 
  delete[] res; 


  return 0;
} 
