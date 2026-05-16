| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 1,385 |  4,000,051 |  424 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 13,340 |  14,365,133 |  2,224 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 8,989 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 1,343 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 472 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 593 |  592,827 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-b11f626c6b1ccafffa05599bf687e30ff6044787.md) | 1,796 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b11f626c6b1ccafffa05599bf687e30ff6044787

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25963693927)
