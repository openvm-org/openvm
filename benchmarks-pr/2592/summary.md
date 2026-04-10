| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 3,889 |  12,000,265 |  968 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 18,725 |  18,655,329 |  3,346 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 9,756 |  14,793,960 |  1,384 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 1,417 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 642 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 920 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 2,151 |  2,579,903 |  437 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 1,717 |  12,000,265 |  456 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 854 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 410 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 518 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.md) | 2,197 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24267614252)
