| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 3,836 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 18,532 |  18,655,329 |  3,322 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 1,437 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 651 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 903 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-9c0103669fe8a5a419c6d99acf2134a41160f1ae.md) | 2,288 |  2,579,903 |  446 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c0103669fe8a5a419c6d99acf2134a41160f1ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24107000605)
