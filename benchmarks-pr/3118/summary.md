| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/fibonacci-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 471 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/keccak-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 7,376 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/sha2_bench-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 4,214 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/regex-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/ecrecover-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 200 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/pairing-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 231 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/kitchen_sink-4777044d441d7ca7dd7792de43212ad9cfc69679.md) | 2,011 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4777044d441d7ca7dd7792de43212ad9cfc69679

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31538701005)
