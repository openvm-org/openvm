| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/fibonacci-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 3,763 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/keccak-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 18,438 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/sha2_bench-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 10,237 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/regex-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 1,399 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/ecrecover-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 605 |  123,583 |  256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/pairing-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 892 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/kitchen_sink-0d918c4f433dec31145eb9cf4c6286c4c0f04bc5.md) | 1,910 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0d918c4f433dec31145eb9cf4c6286c4c0f04bc5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26105238308)
