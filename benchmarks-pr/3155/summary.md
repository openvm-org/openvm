| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 487 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 7,528 |  14,365,133 |  1,617 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 4,337 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 749 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 212 |  112,210 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 253 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 2,261 |  1,979,971 |  479 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci_e2e-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 762 |  4,000,053 |  226 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex_e2e-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 1,080 |  4,090,658 |  207 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover_e2e-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 517 |  112,212 |  180 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing_e2e-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 562 |  592,829 |  164 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink_e2e-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 2,479 |  1,979,973 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b9a90104160de6419b24d7fa976861faa5f63e17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34622979806)
