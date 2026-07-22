| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/fibonacci-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 472 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/keccak-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 7,329 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/sha2_bench-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 4,696 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/regex-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 672 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/ecrecover-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 229 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/pairing-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 322 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/kitchen_sink-7e21a814449b70b2b5690363da37950d5bb397cd.md) | 2,679 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7e21a814449b70b2b5690363da37950d5bb397cd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29952292067)
