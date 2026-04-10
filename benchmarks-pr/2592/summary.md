| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 3,877 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 18,868 |  18,655,329 |  3,377 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 1,425 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 650 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 917 |  1,745,757 |  277 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-90c15af6e34f5e4e5c26ee1ebe3920d45eecf052.md) | 2,155 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90c15af6e34f5e4e5c26ee1ebe3920d45eecf052

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24250900710)
