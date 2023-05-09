# graph_partition
Streaming graph partition written in c++

## Compile
```
mkdir build && cd build
cmake ..
make
```

## Run
```
./run [N (1000 / 5000 / 20000 / 50000)]
```

## Algorithm
<img src="docs/algo.png" width="500">

---

### Parallel Block Merge
To find the optimal proposal block with the smallest change in delta entropy for merging, each block is given K attempts.

<img src="docs/parallel_merge.jpeg" width="700">

---

### Iterative Nodal Update
- Baseline version

<img src="docs/itr_nodal.jpeg" width="700">

### Parallel Nodal Update
- Accept or reject the nodal movement based on the same state, and then update the partition after this run

<img src="docs/parallel_nodal.jpeg" width="700">

### Parallel Batch Nodal Update

- Choose a batch of nodal to move

<img src="docs/batch.jpeg" width="700">

---

### Hybrid Data Structure

<img src="docs/data.jpeg" width="500">

---

### Taskflow

<img src="docs/parallel.jpeg" width="700">

<img src="docs/taskflow.jpeg" width="700">

---

## Result

### Static Graph Statistics
The dataset is from [2022 Streaming Partition Challenge Datasets with Known Truth Partitions](http://graphchallenge.mit.edu/data-sets)

<img src="docs/result_table.jpeg" width="700">

- Using different block size to determine to use matrix or list

<img src="docs/block_size.jpeg" width="700">

## TODO
- [ ] Streaming Graph Input
