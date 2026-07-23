| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 484 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 7,381 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 4,662 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 689 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 226 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 277 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-69f4412471e77f2adea6cfb18ad7769a2a4bbf46.md) | 2,738 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/69f4412471e77f2adea6cfb18ad7769a2a4bbf46

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29969984419)
