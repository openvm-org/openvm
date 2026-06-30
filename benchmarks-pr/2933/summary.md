| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/fibonacci-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 1,043 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/keccak-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 16,228 |  14,365,133 |  3,034 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/sha2_bench-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 8,228 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/regex-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 1,204 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/ecrecover-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 441 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/pairing-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 598 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/kitchen_sink-76a6901f7b46a8ddf81359e7a39182749bd6e467.md) | 3,905 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76a6901f7b46a8ddf81359e7a39182749bd6e467

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28450428969)
