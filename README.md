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
| 20000    | 473914  |                  |  3913162 ms             | |

### Execution Time Breakdown
* 1000
<table>
    <thead>
        <tr>
            <th colspan=3>Block Merge</th>
            <th colspan=3>Nodal Update</th>
            <th>Overall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=3>278 ms</td>
            <td colspan=3>454 ms</td>
            <td>753 ms</td>
        </tr>
        <tr>
            <td>_Propose</td>
            <td>_New_M</td>
            <td>_Delta_Entropy</td>
            <td>_Propose</td>
            <td>_New_M</td>
            <td>_Delta_Entropy</td>
            <td></td>
        </tr>
        <tr>
            <td>33.6 ms</td>
            <td>64.3 ms</td>
            <td>111.7 ms</td>
            <td>23.8 ms</td>
            <td>10.2 ms</td>
            <td>248.3 ms</td>
            <td></td>
        </tr>
        <tr>
            <td colspan=3>52 ms</td>
            <td colspan=3>447 ms</td>
            <td>544 ms</td>
        </tr>
    </tbody>
</table>

* 5000
<table>
    <thead>
        <tr>
            <th colspan=3>Block Merge</th>
            <th colspan=3>Nodal Update</th>
            <th>Overall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=3>13857 ms</td>
            <td colspan=3>8022 ms</td>
            <td>22159 ms</td>
        </tr>
        <tr>
            <td>_Propose</td>
            <td>_New_M</td>
            <td>_Delta_Entropy</td>
            <td>_Propose</td>
            <td>_New_M</td>
            <td>_Delta_Entropy</td>
            <td></td>
        </tr>
        <tr>
            <td>785.4 ms</td>
            <td>5266.9 ms</td>
            <td>5950.0 ms</td>
            <td>455.9 ms</td>
            <td>1903.4 ms</td>
            <td>4429.7 ms</td>
            <td></td>
        </tr>
        <tr>
            <td colspan=3>1601 ms</td>
            <td colspan=3>8310 ms</td>
            <td>10534 ms</td>
        </tr>
    </tbody>
</table>

## TODO
- [ ] Streaming Graph Input
