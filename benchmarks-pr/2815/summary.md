| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/fibonacci-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 3,762 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/keccak-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 18,670 |  18,655,329 |  3,293 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/sha2_bench-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 10,140 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/regex-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 1,403 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/ecrecover-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 603 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/pairing-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 894 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/kitchen_sink-31f809b5c13eba9a403ac86b5451c8313b61f118.md) | 1,889 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/31f809b5c13eba9a403ac86b5451c8313b61f118

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26647728005)
