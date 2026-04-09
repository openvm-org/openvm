| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/fibonacci-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 3,773 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/keccak-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 18,704 |  18,655,329 |  3,348 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/regex-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 1,429 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/ecrecover-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/pairing-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 908 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2688/kitchen_sink-f3c45d7c54e31b100ebf9f7584b0539ad9b03538.md) | 2,148 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f3c45d7c54e31b100ebf9f7584b0539ad9b03538

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24212097835)
