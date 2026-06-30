| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/fibonacci-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 865 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/keccak-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 15,387 |  14,365,133 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/sha2_bench-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 8,058 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/regex-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 1,030 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/ecrecover-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 300 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/pairing-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 448 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/kitchen_sink-f6b197b102fc36cf6035c15e34c9efde40d1a3ec.md) | 3,746 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f6b197b102fc36cf6035c15e34c9efde40d1a3ec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28479551107)
