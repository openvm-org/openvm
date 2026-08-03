| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/fibonacci-bbd05990f982645c80a2b88595399c1809216bea.md) | 480 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/keccak-bbd05990f982645c80a2b88595399c1809216bea.md) | 7,391 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/sha2_bench-bbd05990f982645c80a2b88595399c1809216bea.md) | 4,168 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/regex-bbd05990f982645c80a2b88595399c1809216bea.md) | 646 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/ecrecover-bbd05990f982645c80a2b88595399c1809216bea.md) | 230 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/pairing-bbd05990f982645c80a2b88595399c1809216bea.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/kitchen_sink-bbd05990f982645c80a2b88595399c1809216bea.md) | 2,043 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bbd05990f982645c80a2b88595399c1809216bea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30827159452)
