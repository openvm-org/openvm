| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 973 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 15,846 |  14,365,133 |  3,055 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 8,189 |  11,167,961 |  1,011 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 1,176 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 437 |  112,210 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 582 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-3b28e8850e64806a44a29fce4f911808c41aa571.md) | 3,796 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b28e8850e64806a44a29fce4f911808c41aa571

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28763731834)
