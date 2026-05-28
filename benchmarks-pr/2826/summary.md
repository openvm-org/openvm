| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 3,810 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 18,397 |  18,655,329 |  3,241 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 10,331 |  14,793,960 |  1,476 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 1,397 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 604 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 886 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-ae25f0ec8b2ed49897365295aa67de838e2b90e8.md) | 1,891 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ae25f0ec8b2ed49897365295aa67de838e2b90e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26607527168)
