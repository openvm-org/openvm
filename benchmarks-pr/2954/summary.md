| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/fibonacci-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 3,010 |  12,000,265 |  662 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/keccak-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 16,288 |  18,655,329 |  3,022 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/sha2_bench-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 9,128 |  14,793,960 |  1,111 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/regex-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 1,158 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/ecrecover-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 602 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/pairing-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 935 |  1,745,757 |  310 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/kitchen_sink-401eed358e3ff786a94ecef2fce9b45da07b54b4.md) | 4,092 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/401eed358e3ff786a94ecef2fce9b45da07b54b4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28556427289)
