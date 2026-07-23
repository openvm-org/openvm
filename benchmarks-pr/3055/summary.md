| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-039c270bf8265189191928f59f8afca8710b7d9d.md) | 465 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-039c270bf8265189191928f59f8afca8710b7d9d.md) | 7,340 |  14,365,133 |  1,553 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-039c270bf8265189191928f59f8afca8710b7d9d.md) | 4,786 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-039c270bf8265189191928f59f8afca8710b7d9d.md) | 681 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-039c270bf8265189191928f59f8afca8710b7d9d.md) | 226 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-039c270bf8265189191928f59f8afca8710b7d9d.md) | 318 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-039c270bf8265189191928f59f8afca8710b7d9d.md) | 2,661 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/039c270bf8265189191928f59f8afca8710b7d9d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30024196234)
