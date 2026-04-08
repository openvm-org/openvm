| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/fibonacci-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 3,787 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/keccak-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 18,386 |  18,655,329 |  3,294 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/regex-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 1,415 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/ecrecover-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 642 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/pairing-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 897 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/kitchen_sink-8f1936e6d58306f80b7d2e226f3acd2726e36b59.md) | 2,156 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8f1936e6d58306f80b7d2e226f3acd2726e36b59

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24146529044)
