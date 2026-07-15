| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 414 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 8,460 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 4,130 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 497 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 218 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 277 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d.md) | 1,997 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0d0a67712fc6d9c62f3240e4f47cea8a9bfc4a7d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29446827025)
