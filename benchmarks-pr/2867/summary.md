| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/fibonacci-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 4,013 |  12,000,265 |  1,151 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/keccak-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 21,764 |  18,655,329 |  4,601 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/sha2_bench-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 9,672 |  14,793,960 |  1,857 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/regex-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 1,508 |  4,137,067 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/ecrecover-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 606 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/pairing-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 931 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2867/kitchen_sink-d7b8b75608451b52ebcdedce16681dcbbe5ed5e8.md) | 4,127 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d7b8b75608451b52ebcdedce16681dcbbe5ed5e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27285224761)
