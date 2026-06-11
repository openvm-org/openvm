| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/fibonacci-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 4,015 |  12,000,265 |  1,152 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/keccak-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 21,862 |  18,655,329 |  4,646 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/sha2_bench-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 9,524 |  14,793,960 |  1,835 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/regex-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 1,518 |  4,137,067 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/ecrecover-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 615 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/pairing-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 952 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/kitchen_sink-4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e.md) | 4,140 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4cfd2b91eeb7ce3fef5e28984d8b13028d73ea5e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27362246731)
