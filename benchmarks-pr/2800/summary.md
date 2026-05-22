| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/fibonacci-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 3,738 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/keccak-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 18,621 |  18,655,329 |  3,276 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/sha2_bench-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 10,266 |  14,793,960 |  1,468 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/regex-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 1,395 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/ecrecover-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 607 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/pairing-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 892 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/kitchen_sink-d7ab381f8b9c3518b7bca8f0cf3de7f52e662017.md) | 1,909 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d7ab381f8b9c3518b7bca8f0cf3de7f52e662017

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26260249944)
