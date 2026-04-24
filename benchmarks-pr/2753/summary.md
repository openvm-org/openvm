| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/fibonacci-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 3,870 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/keccak-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 18,580 |  18,655,329 |  3,307 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/sha2_bench-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 9,029 |  14,793,960 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/regex-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 1,433 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/ecrecover-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 642 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/pairing-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 902 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2753/kitchen_sink-0c3ad8c7da03356709e161b52b6413f6d0519ef7.md) | 2,090 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c3ad8c7da03356709e161b52b6413f6d0519ef7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24913936920)
