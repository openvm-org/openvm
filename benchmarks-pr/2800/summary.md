| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/fibonacci-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 3,790 |  12,000,265 |  927 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/keccak-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 18,534 |  18,655,329 |  3,262 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/sha2_bench-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 10,121 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/regex-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 1,395 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/ecrecover-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 603 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/pairing-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 896 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/kitchen_sink-18dfbf6e758315abe7a932ea06c5ed6e93159289.md) | 1,901 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/18dfbf6e758315abe7a932ea06c5ed6e93159289

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26190686433)
