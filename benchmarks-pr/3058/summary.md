| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 471 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 7,321 |  14,365,133 |  1,555 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 4,736 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 666 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 224 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 328 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-5f9e1672bfc29973172ccaf3fd379f91de7fa7e2.md) | 2,617 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5f9e1672bfc29973172ccaf3fd379f91de7fa7e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29873005389)
