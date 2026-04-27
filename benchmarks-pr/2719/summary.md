| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 1,874 |  4,000,051 |  532 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 13,734 |  14,365,133 |  2,259 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 9,356 |  11,167,961 |  1,246 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 1,580 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 648 |  112,210 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 762 |  592,827 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-a6bc526bdc6d55c2c35079db205610c2c66d1a39.md) | 2,084 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a6bc526bdc6d55c2c35079db205610c2c66d1a39

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25020083450)
