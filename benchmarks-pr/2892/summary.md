| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/fibonacci-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 1,699 |  4,000,051 |  547 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/keccak-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 16,301 |  14,365,133 |  3,016 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/sha2_bench-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 10,397 |  11,167,961 |  1,930 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/regex-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 1,552 |  4,090,656 |  436 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/ecrecover-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 478 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/pairing-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 618 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/kitchen_sink-c9512c81565ca7a0716b6a94ad2b53f559cbca8c.md) | 3,972 |  1,979,971 |  869 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c9512c81565ca7a0716b6a94ad2b53f559cbca8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27709647748)
