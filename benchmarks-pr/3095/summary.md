| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/fibonacci-a071275370de991f8822fedc25b759689fb00395.md) | 476 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/keccak-a071275370de991f8822fedc25b759689fb00395.md) | 7,348 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/sha2_bench-a071275370de991f8822fedc25b759689fb00395.md) | 4,153 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/regex-a071275370de991f8822fedc25b759689fb00395.md) | 655 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/ecrecover-a071275370de991f8822fedc25b759689fb00395.md) | 225 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/pairing-a071275370de991f8822fedc25b759689fb00395.md) | 231 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/kitchen_sink-a071275370de991f8822fedc25b759689fb00395.md) | 2,038 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a071275370de991f8822fedc25b759689fb00395

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30919470231)
