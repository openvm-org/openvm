| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/fibonacci-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 3,761 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/keccak-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 18,754 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/sha2_bench-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 10,111 |  14,793,960 |  1,448 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/regex-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 1,418 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/ecrecover-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 597 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/pairing-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 882 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/kitchen_sink-bbe12a2c7f6eafac19aeb33adaa24c00763da786.md) | 1,900 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bbe12a2c7f6eafac19aeb33adaa24c00763da786

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26283607883)
