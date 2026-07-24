| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 471 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 7,344 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 4,694 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 667 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 229 |  112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 320 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-75da77e6eae02a3a5886063fde28f35f996791b2.md) | 2,669 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/75da77e6eae02a3a5886063fde28f35f996791b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30132925941)
