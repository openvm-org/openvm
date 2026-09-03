| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/fibonacci-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 486 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/keccak-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 7,703 |  14,365,133 |  1,654 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/sha2_bench-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 4,366 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/regex-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 783 |  4,090,656 |  223 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/ecrecover-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 208 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/pairing-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 251 |  592,827 |  171 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/kitchen_sink-d926a5bdff638f9f86875e2774fcb9860e4964e5.md) | 2,250 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d926a5bdff638f9f86875e2774fcb9860e4964e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33815848759)
