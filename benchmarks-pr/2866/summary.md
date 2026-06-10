| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/fibonacci-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 4,062 |  12,000,265 |  1,168 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/keccak-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 21,777 |  18,655,329 |  4,620 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/sha2_bench-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 9,651 |  14,793,960 |  1,841 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/regex-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 1,490 |  4,137,067 |  426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/ecrecover-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 606 |  123,583 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/pairing-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 938 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/kitchen_sink-8338ed718ad7c0bce30b69b917771c0dabdf2db9.md) | 4,156 |  2,579,903 |  891 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8338ed718ad7c0bce30b69b917771c0dabdf2db9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27286548434)
