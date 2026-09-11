| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 477 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 7,686 |  14,365,133 |  1,631 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 4,287 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 754 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 208 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 252 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-b9a90104160de6419b24d7fa976861faa5f63e17.md) | 2,242 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b9a90104160de6419b24d7fa976861faa5f63e17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34620213219)
