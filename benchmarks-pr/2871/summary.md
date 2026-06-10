| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/fibonacci-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 3,997 |  12,000,265 |  1,148 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/keccak-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 21,611 |  18,655,329 |  4,593 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/sha2_bench-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 9,561 |  14,793,960 |  1,837 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/regex-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 1,520 |  4,137,067 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/ecrecover-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 614 |  123,583 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/pairing-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 946 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/kitchen_sink-9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb.md) | 4,141 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c5ddf4c5ee71261ddf4ba9f3772e1fb62e88bbb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27313968934)
