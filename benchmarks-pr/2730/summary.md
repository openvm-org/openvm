| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/fibonacci-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 3,823 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/keccak-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 18,666 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/sha2_bench-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 8,981 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/regex-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 1,429 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/ecrecover-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 647 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/pairing-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 906 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/kitchen_sink-f114146deb87eec32179d3163d73a93b67f77d8c.md) | 2,118 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f114146deb87eec32179d3163d73a93b67f77d8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24809161234)
