| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 413 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 8,331 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 4,139 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 500 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 219 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 264 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-e360e227227eb7acbc525b622b7bd3fb77565ad9.md) | 2,011 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e360e227227eb7acbc525b622b7bd3fb77565ad9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29435238766)
