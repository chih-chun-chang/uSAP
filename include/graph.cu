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
      odata[i] += idata[i*width + j];
    }
  }

}

void matrix_row_divide_seq(float* odata, int* idata1, int* idata2, unsigned width) {

  for (unsigned i = 0; i < width; i++) {
	for (unsigned j = 0; j < width; j++) {
	  odata[i*width + j] = (float)idata1[i*width + j]/idata2[i];
	}
  }

}

template <typename T>
void check(T* odata, T* idata, unsigned width) {
  for (unsigned i = 0; i < width; i++) { 
    for (unsigned j = 0; j < width; j++) {
      if (odata[i*width + j] != idata[i*width + j]) {
        std::cout << odata[i*width + j] << ", " << idata[i*width + j] << "\n";
        std::cout << i << ", " << j << "\n";
		assert(odata[i*width + j] == idata[i*width + j]);
      }
    }
  }
} 

void check1d(int* odata, int* idata, unsigned width) {
  for (unsigned i = 0; i < width; i++) {
    if (odata[i] != idata[i]) {
      std::cout << odata[i] << ", " << idata[i] << "\n";
      std::cout << i << "\n";
	  assert(odata[i] == idata[i]);
    }
  }
}

// ----------------------CUDA kernel ---------------------//


__global__ void matrix_transpose(int* odata, int* idata, unsigned width) {

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

__global__ void vector_addition(int* odata, int* idata1, int* idata2, unsigned width) {

  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x < width) {
	odata[x] = idata1[x] + idata2[x];
  }

}


__global__ void matrix_row_reduce(int* odata, int* idata, unsigned width) {

  // thread blocks of dimension 32x8
  // each block copiesa tile of dimension 32x32
  // each thread responsible for 32/8 data

  __shared__ int tile[TILE_DIM][TILE_DIM+1];

  unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
  unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;
  unsigned index = x + y*width;


  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
	tile[threadIdx.y + i][threadIdx.x] = idata[index + i*width];
  }

  __syncthreads();

  for (unsigned s = TILE_DIM / 2; s > 0; s>>=1) {
    if (threadIdx.x < s) {
      for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] += tile[threadIdx.y + i][threadIdx.x + s];
      }
    }
    __syncthreads();
  }


  if (threadIdx.x == 0) {
	for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
      atomicAdd(&odata[y+i], tile[threadIdx.y + i][0]);
	}
  }

}

__global__ void matrix_row_divide(float* odata, int* idata1, int* idata2, unsigned width) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
  unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;
  unsigned index = x + y*width;

  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.x][threadIdx.y + i] = (float)idata1[index + i*width]/idata2[y + i];
  }

  __syncthreads();

  for (unsigned i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
    odata[index + i*width] = tile[threadIdx.x][threadIdx.y + i];
  }

}

__global__ void matrix_discrete_distribution(int* odata, float* P, float* rand, unsigned width) {

  unsigned row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < width) {
	
	float cumulativeProbability = 0.0;
    float randValue = rand[row];
	
	for (unsigned col = 0; col < width; col++) {
	  cumulativeProbability += P[row * width + col];
	  if (randValue <= cumulativeProbability) {
		odata[row] = col;
        break;
	  }
	}	
  }
}

__global__ void vector_uniform_number_generator(float* odata, unsigned width) {

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < width) {
	curandState localState;
    curand_init(clock64() + idx, 0, 0, &localState);
    odata[idx] = curand_uniform(&localState);
  }

}


__global__ void vector_random_number_generator(int* odata, int* idata, unsigned width) {

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < width && idata[idx] == 0) {
	curandState localState;
    curand_init(clock64() + idx, 0, 0, &localState);
	int rand = curand(&localState) % width;
    if (rand == idx) rand++;
    if (rand == width) rand = 0;
    odata[idx] = rand;
  }
}

// calculate B/(d[u]+B)
__global__ void calculate_prob(float* odata, int* d, unsigned width) {

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < width) {
	odata[idx] = (float)width/(d[idx]+width);
  }

}

__global__ void vector_map(int* odata, int* idata, unsigned* index, unsigned width) {

  unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (idx < width) {
    odata[idx] = idata[index];
  }

}
   


// ---------------------- Partition -------------------------//
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

  Node* S; // proposal for each block (B*1)
  unsigned* n; // random index (B*1)
  Node* dG;
  Node* u; 

  // TODO: malloc only once
  //11.2
  //cudaMallocAsync(&dout, sizeof(int)*B, stream);
  //cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width*width, cudaMemcpyDefault, stream);
  cudaMalloc(&gpu_M, sizeof(Weight)*B*B);
  cudaMalloc(&gpu_Mt, sizeof(Weight)*B*B);
  cudaMalloc(&dout, sizeof(Weight)*B);
  cudaMalloc(&din, sizeof(Weight)*B);
  cudaMalloc(&d, sizeof(Weight)*B);
  cudaMalloc(&P, sizeof(float)*B*B);
  cudaMalloc(&S, sizeof(Node)*B);
  cudaMalloc(&n, sizeof(unsigned)*B);
  cudaMalloc(&dG, sizeof(Node)*B);
  cudaMalloc(&u, sizeof(Node)*B);

  cudaMemcpy(gpu_M, M, sizeof(Weight)*B*B, cudaMemcpyDefault);
  cudaMemcpy(dG, G, sizeof(Node)*B, cudaMemcpyDefault);

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
  // 1.   sample an index n based on P
  // 2.   u = G[n]
  // 3.   x ~ U(0,1)
  // 4.   prob = B/(u+B)
  // 5.   x <= prob[u] -> S[i] = rand(B)
  // 6.   x > prob[u] ->
  //        rowsum(prob) == prob[i] -> S[i] = rand(B)
  //        prob / rowsum(prob) -> S[i] = sample
  matrix_discrete_distribution<<<1, B, 0, stream>>>(
    // TODO:...
  ); 
  

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
  float* cpu_out2 = new float[width*width];
  matrix_row_reduce_seq(cpu_out, M.data(), width);
  matrix_row_divide_seq(cpu_out2, M.data(), cpu_out, width);

  //////////////
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int* gpu_in;
  int* gpu_out;
  float* gpu_out2;
  float* res = new float[width*width];
  //int* res = new int[width];

  // 11.2
  //cudaMallocAsync(&gpu_in, sizeof(int)*width*width, stream);
  //cudaMallocAsync(&gpu_out, sizeof(int)*width*width, stream);
  //cudaMallocAsync(&gpu_out, sizeof(int)*width, stream);

  //cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width*width, cudaMemcpyDefault, stream);
  //cudaMemcpyAsync(gpu_in, M.data(), sizeof(int)*width, cudaMemcpyDefault, stream);


  // 11.1
  cudaMalloc(&gpu_in, sizeof(int)*width*width);
  cudaMalloc(&gpu_out, sizeof(int)*width);
  cudaMalloc(&gpu_out2, sizeof(int)*width*width);
  cudaMemcpy(gpu_in, M.data(), sizeof(int)*width*width, cudaMemcpyDefault);


  //matrix_transposeDiagonal<<<dimGrid, dimBlock, 0, stream>>>(
  //  gpu_out, gpu_in, width
  //);
  //matrix_addition<<<dimGrid, dimBlock, 0, stream>>>(
  //  gpu_out, gpu_out, gpu_in, width
  //);
  matrix_row_reduce<<<dimGrid, dimBlock, 0, stream>>>(
    gpu_out, gpu_in, width
  );
  matrix_row_divide<<<dimGrid, dimBlock, 0, stream>>>(
	gpu_out2, gpu_in, gpu_out, width
  );

  // 11.2
  //cudaMemcpyAsync(res, gpu_out, sizeof(int)*width*width, cudaMemcpyDefault, stream);
  //cudaMemcpyAsync(res, gpu_out, sizeof(int)*width, cudaMemcpyDefault, stream);

  cudaMemcpy(res, gpu_out2, sizeof(int)*width*width, cudaMemcpyDefault);

  cudaStreamSynchronize(stream);

  ////////////


  check<float>(cpu_out2, res, width);
  //check1d(cpu_out, res, width);


  ///////////
  //cudaFreeAsync(gpu_in, stream);
  //cudaFreeAsync(gpu_out, stream);
  cudaStreamDestroy(stream);

  delete[] cpu_out; 
  delete[] res; 


  return 0;
} 
