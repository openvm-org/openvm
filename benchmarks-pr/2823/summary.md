| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/fibonacci-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 3,728 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/keccak-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 18,467 |  18,655,329 |  3,261 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/sha2_bench-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 10,142 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/regex-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 1,395 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/ecrecover-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 614 |  123,583 |  260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/pairing-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 894 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/kitchen_sink-7fe5c6e1ea10c9830513116bd38906c5e83f08ac.md) | 1,903 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7fe5c6e1ea10c9830513116bd38906c5e83f08ac

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26650330026)
