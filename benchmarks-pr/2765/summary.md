| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 1,880 |  4,000,051 |  515 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 13,469 |  14,365,133 |  2,203 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 9,451 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 1,573 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 598 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 734 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-8e89075b1e830c7733ba932e43e6c1632087478a.md) | 1,868 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e89075b1e830c7733ba932e43e6c1632087478a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26244973519)
