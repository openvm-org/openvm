| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 1,901 |  4,000,051 |  538 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 13,660 |  14,365,133 |  2,251 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 9,462 |  11,167,961 |  1,416 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 1,603 |  4,090,656 |  383 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 636 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 754 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-6659be49555b8a2987f28818cf4f08a2cd547dc9.md) | 2,049 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6659be49555b8a2987f28818cf4f08a2cd547dc9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25854161116)
