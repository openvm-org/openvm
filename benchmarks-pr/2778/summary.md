| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-077b9080da1e395662136d021e664301503372ca.md) | 1,413 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-077b9080da1e395662136d021e664301503372ca.md) | 13,266 |  14,365,133 |  2,188 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-077b9080da1e395662136d021e664301503372ca.md) | 9,061 |  11,167,961 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-077b9080da1e395662136d021e664301503372ca.md) | 1,357 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-077b9080da1e395662136d021e664301503372ca.md) | 472 |  112,210 |  261 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-077b9080da1e395662136d021e664301503372ca.md) | 587 |  592,827 |  246 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-077b9080da1e395662136d021e664301503372ca.md) | 1,788 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/077b9080da1e395662136d021e664301503372ca

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25938290762)
