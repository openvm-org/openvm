| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 479 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 7,350 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 4,188 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 671 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc.md) | 2,042 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/764a8e0d8be5a0e3a1e52cda3206ae308b28e3cc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31438717366)
