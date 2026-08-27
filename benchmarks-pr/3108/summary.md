| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 462 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 7,425 |  14,365,133 |  1,640 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 4,120 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 732 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 205 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 243 |  592,827 |  170 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-d9908ce79abc31c4adc6a3b928f2c1c836840c5c.md) | 2,131 |  1,979,971 |  453 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d9908ce79abc31c4adc6a3b928f2c1c836840c5c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33123104952)
