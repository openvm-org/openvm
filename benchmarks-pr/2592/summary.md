| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 3,848 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 18,402 |  18,655,329 |  3,258 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 1,451 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 643 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 902 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-f2c28eceef5c26c27c770950cd53e60fdec2b766.md) | 2,278 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f2c28eceef5c26c27c770950cd53e60fdec2b766

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23614503676)
