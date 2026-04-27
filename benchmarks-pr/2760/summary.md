| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/fibonacci-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 1,900 |  4,000,051 |  534 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/keccak-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 13,613 |  14,365,133 |  2,243 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/sha2_bench-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 9,372 |  11,167,961 |  1,262 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/regex-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 1,586 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/ecrecover-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 649 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/pairing-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 766 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/kitchen_sink-a2d264276a435bf3aa23a1dd5e332e3413e982ba.md) | 2,070 |  1,979,971 |  428 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2d264276a435bf3aa23a1dd5e332e3413e982ba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25020064015)
