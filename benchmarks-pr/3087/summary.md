| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/fibonacci-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 452 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/keccak-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 7,238 |  14,365,133 |  1,631 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/sha2_bench-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 4,630 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/regex-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 647 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/ecrecover-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/pairing-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 309 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/kitchen_sink-77b9e0d053bf5d143c821751d97f3f0ec5ca5b60.md) | 2,597 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/77b9e0d053bf5d143c821751d97f3f0ec5ca5b60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30650289139)
