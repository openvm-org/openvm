| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/fibonacci-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 477 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/keccak-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 7,346 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/sha2_bench-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 4,158 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/regex-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 657 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/ecrecover-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 230 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/pairing-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/kitchen_sink-422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9.md) | 2,026 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/422b5a0d21cddb1fdb3de7bf71ffdecff3824ee9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30886941569)
