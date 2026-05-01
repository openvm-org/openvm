| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/fibonacci-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 3,793 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/keccak-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 18,604 |  18,655,329 |  3,332 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/sha2_bench-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 8,873 |  14,793,960 |  1,378 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/regex-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 1,414 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/ecrecover-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 653 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/pairing-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 895 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/kitchen_sink-07d636e2883af2c2781f2d6214f433db1d41d302.md) | 2,086 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07d636e2883af2c2781f2d6214f433db1d41d302

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25199271679)
