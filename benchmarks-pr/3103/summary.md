| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-989fac1962700df0ff172178945af37d291640d7.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-989fac1962700df0ff172178945af37d291640d7.md) | 7,313 |  14,365,133 |  1,506 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-989fac1962700df0ff172178945af37d291640d7.md) | 4,166 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-989fac1962700df0ff172178945af37d291640d7.md) | 665 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-989fac1962700df0ff172178945af37d291640d7.md) | 197 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-989fac1962700df0ff172178945af37d291640d7.md) | 235 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-989fac1962700df0ff172178945af37d291640d7.md) | 2,039 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/989fac1962700df0ff172178945af37d291640d7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32177943254)
