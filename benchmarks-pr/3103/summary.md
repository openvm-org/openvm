| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-d33f5fe32393420828b951586707f5d84bc57009.md) | 474 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-d33f5fe32393420828b951586707f5d84bc57009.md) | 7,399 |  14,365,133 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-d33f5fe32393420828b951586707f5d84bc57009.md) | 4,138 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-d33f5fe32393420828b951586707f5d84bc57009.md) | 661 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-d33f5fe32393420828b951586707f5d84bc57009.md) | 223 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-d33f5fe32393420828b951586707f5d84bc57009.md) | 228 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-d33f5fe32393420828b951586707f5d84bc57009.md) | 2,025 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d33f5fe32393420828b951586707f5d84bc57009

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31219302452)
