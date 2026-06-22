| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/fibonacci-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 3,071 |  12,000,265 |  675 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/keccak-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 16,363 |  18,655,329 |  3,038 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/sha2_bench-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 9,115 |  14,793,960 |  1,113 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/regex-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 1,166 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/ecrecover-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 605 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/pairing-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 940 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2924/kitchen_sink-6ce44dd8306a712553030d4df07520eb7edc7ff5.md) | 4,145 |  2,579,903 |  889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6ce44dd8306a712553030d4df07520eb7edc7ff5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27982542155)
