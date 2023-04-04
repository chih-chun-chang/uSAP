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
| 20000    | 473914  |||
| 50000    | 1189382 |||
| 200000   | |||
| 1000000  | |||
| 5000000  | |||
| 20000000 | |||

## TODO
- [ ] Parallel Blockwise Merge
- [ ] Streaming Graph Input
