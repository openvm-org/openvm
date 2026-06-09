| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 3,975 |  12,000,265 |  1,154 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 21,753 |  18,655,329 |  4,599 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 9,676 |  14,793,960 |  1,849 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 1,487 |  4,137,067 |  425 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 605 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 955 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-1c5378a641fd3a41a163be6a214b3962a4af3104.md) | 4,188 |  2,579,903 |  894 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1c5378a641fd3a41a163be6a214b3962a4af3104

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27223740011)
