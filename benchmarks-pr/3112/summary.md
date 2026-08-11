| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 467 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 7,420 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 4,186 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 671 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 202 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-19920d8440a309c74d4bb480a24f89d27907cb5f.md) | 2,044 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19920d8440a309c74d4bb480a24f89d27907cb5f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31527005109)
