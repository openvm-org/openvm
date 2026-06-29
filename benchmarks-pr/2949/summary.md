| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/fibonacci-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 1,032 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/keccak-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 15,822 |  14,365,133 |  3,052 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/sha2_bench-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 8,183 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/regex-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 1,165 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/ecrecover-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 445 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/pairing-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 586 |  592,827 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/kitchen_sink-0ffa10185515693c59651c1b992c123fb91cd3c0.md) | 3,817 |  1,979,971 |  853 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0ffa10185515693c59651c1b992c123fb91cd3c0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28406375732)
