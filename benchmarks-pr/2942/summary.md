| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/fibonacci-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 856 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/keccak-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 15,381 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/sha2_bench-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 8,097 |  11,167,961 |  1,015 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/regex-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 1,046 |  4,090,656 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/ecrecover-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 303 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/pairing-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 444 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/kitchen_sink-919268aeef370b65a8aab13c9b3eefb5f401dbbf.md) | 3,705 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/919268aeef370b65a8aab13c9b3eefb5f401dbbf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28454520735)
