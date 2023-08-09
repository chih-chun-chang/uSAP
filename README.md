# uSAP: An Ultra-Fast Stochastic Graph Partitioner
Streaming graph partition written in c++

## Compile
```
mkdir build && cd build
cmake ..
make
```

## Run
```
./run  [ (1 / 2 / 3 / 4) ] [N (1000 / 5000 / 20000 / 50000)]
```

## Algorithm

---

### Parallel Block Merge
To find the optimal proposal block with the smallest change in delta entropy for merging, each block is given K attempts.

<img src="docs/parallel_merge.jpeg" width="600">

---

### Parallel Batch Nodal Update

- Choose a batch of nodal to move

<img src="docs/batch.jpg" width="700">

---

### Taskflow

<img src="docs/taskgraph.jpg" width="500">

---

## Result

The dataset is from [2022 Streaming Partition Challenge Datasets with Known Truth Partitions](http://graphchallenge.mit.edu/data-sets)

<img src="docs/result.png" width="700">

## TODO
- [ ] Streaming Graph Input
- [ ] GPU

## GPU version

### Data Structure

<img src="docs/gpu_data.jpg" width="600">

### Block Merge

- Sequential flow

<img src="docs/merge_seq_flow.jpg" width="500">

- Parallel Using ```CudaStream_t```

<img src="docs/merge_flow_stream.jpg" width="700">

### Compute the Change of Entropy

<img src="docs/ds_gpu.jpg" width="700">

- Entropy after merging

<img src="docs/ds_new.jpg" width="600">
