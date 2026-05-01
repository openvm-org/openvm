| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/fibonacci-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 3,847 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/keccak-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 18,828 |  18,655,329 |  3,368 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/sha2_bench-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 9,030 |  14,793,960 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/regex-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 1,414 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/ecrecover-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 635 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/pairing-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 896 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/kitchen_sink-acf9d226bb796675be5efb82d5e7af99639033ca.md) | 2,087 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/acf9d226bb796675be5efb82d5e7af99639033ca

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25236813000)
