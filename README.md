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
| Vertices | Edges   | Truth Partitions | Sequential Exe. Time | Parallel Block Merge (taskflow) |
| -------- | -----   | ---------------- | ------------------ | ---------------|
| 1000     | 8067    | 11               |  711 ms                   | 525 ms |
| 5000     | 50850   | 19               |  22482 ms                 | 10412 |
| 20000    | 473914  |                  |                           | |

## TODO
- [ ] Streaming Graph Input
