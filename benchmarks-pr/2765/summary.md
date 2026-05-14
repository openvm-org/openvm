| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 1,883 |  4,000,051 |  532 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 13,638 |  14,365,133 |  2,253 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 9,467 |  11,167,961 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 1,628 |  4,090,656 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 647 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 753 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-65457db92b207f62706ba6bbbacf7b029a16bea3.md) | 2,024 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/65457db92b207f62706ba6bbbacf7b029a16bea3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25856894628)
