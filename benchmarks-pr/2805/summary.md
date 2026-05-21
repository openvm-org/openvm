| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/fibonacci-513501d09176f2d9e107d153b062b232575fd262.md) | 1,566 |  4,000,051 |  439 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/keccak-513501d09176f2d9e107d153b062b232575fd262.md) | 13,874 |  14,365,133 |  2,373 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/sha2_bench-513501d09176f2d9e107d153b062b232575fd262.md) | 9,233 |  11,167,961 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/regex-513501d09176f2d9e107d153b062b232575fd262.md) | 1,446 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/ecrecover-513501d09176f2d9e107d153b062b232575fd262.md) | 472 |  112,210 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/pairing-513501d09176f2d9e107d153b062b232575fd262.md) | 588 |  592,827 |  254 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/kitchen_sink-513501d09176f2d9e107d153b062b232575fd262.md) | 2,174 |  1,979,971 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/513501d09176f2d9e107d153b062b232575fd262

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26253364296)
