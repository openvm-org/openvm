| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-5291550ffe5f75baf84421021955fc65375e21b7.md) | 472 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-5291550ffe5f75baf84421021955fc65375e21b7.md) | 7,407 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-5291550ffe5f75baf84421021955fc65375e21b7.md) | 4,218 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-5291550ffe5f75baf84421021955fc65375e21b7.md) | 679 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-5291550ffe5f75baf84421021955fc65375e21b7.md) | 197 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-5291550ffe5f75baf84421021955fc65375e21b7.md) | 230 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-5291550ffe5f75baf84421021955fc65375e21b7.md) | 2,052 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5291550ffe5f75baf84421021955fc65375e21b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31549913428)
