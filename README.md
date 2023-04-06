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

## Result

### Static Graph Statistics
The dataset is from [2022 Streaming Partition Challenge Datasets with Known Truth Partitions](http://graphchallenge.mit.edu/data-sets)
| Vertices | Edges   | Truth Partitions | Execution Time (ms) |
| -------- | -----   | ---------------- | ------------------  |
| 1000     | 8067    | 11               |  3209               |
| 5000     | 50850   | 19               |  72419              |

## TODO
- [ ] Parallel Blockwise Merge (Taskflow)
- [ ] Streaming Graph Input
