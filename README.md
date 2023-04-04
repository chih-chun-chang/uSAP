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
./run [Path to dataset tsv file]
```

## Result

### Static Graph Statistics
The dataset is from [2022 Streaming Partition Challenge Datasets with Known Truth Partitions](http://graphchallenge.mit.edu/data-sets)
| Vertices | Edges   | Truth Partitions | Execution Time (ms) |
| -------- | -----   | ---------------- | ------------------  |
| 1000     | 8067    | 11               |  3209               |
| 5000     | 50850   | 19               |  72419              |

* profile of 1000 vertice

| % time | function name |
| ------ | ------------- |
| 20.83  | _compute_new_rows_cols_interblock_edge_count |
| 17.19  | _compute_delta_entropy |
| 9.11   | std::vector<int, std::allocator<int> >::operator[](unsigned long) const |
| 4.17   | _propose_new_partition |
| 2.60   | __gnu_cxx::__normal_iterator<double*, std::vector<double, std::allocator<double> > >::operator++() |
| 2.60   | __gnu_cxx::__enable_if<std::__is_scalar<int>::__value, void>::__type std::__fill_a1<int*, int>(int*, int*, int const&) |
| 2.34   | std::allocator<bool>::allocator() |
| 2.08   | double* std::__copy_move<false, false, std::random_access_iterator_tag>::__copy_m<float*, double*>(float*, float*, double*) |
| 2.08   | partition() |
| 1.82   | std::vector<int, std::allocator<int> >::push_back(int const&) |
| 1.30   | std::vector<int, std::allocator<int> >::operator[](unsigned long) |

* profile of 5000 vertice

| % time | function name |
| ------ | ------------- |
| 41.16  | _compute_new_rows_cols_interblock_edge_count |
| 25.76  | _compute_delta_entropy |
| 6.42   | partition() |
| 5.82   | std::vector<int, std::allocator<int> >::operator[](unsigned long) const |
| 3.02   | __gnu_cxx::__enable_if<std::__is_scalar<int>::__value, void>::__type std::__fill_a1<int*, int>(int*, int*, int const&) |
| 1.97   | _propose_new_partition |
| 1.72   | std::vector<int, std::allocator<int> >::operator[](unsigned long) |
| 1.61   | std::allocator<bool>::allocator() |

* compare block merge and nodal update

| Vertices | Block Merge | Nodal Update | Overall |
| -------  | ----------  | -----------  | ------- |
|1000| 1250 | 1882 | 3179 |
| 5000 | 41841 | 29050 | 71759 | 

## TODO
- [ ] Parallel Blockwise Merge
- [ ] Streaming Graph Input
