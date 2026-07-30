| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 476 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 7,363 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 4,142 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 659 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 231 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 240 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-150229e517b5bc415c35b7c088eccfd9acf81b68.md) | 2,052 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/150229e517b5bc415c35b7c088eccfd9acf81b68

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30521773118)
