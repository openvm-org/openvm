| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/fibonacci-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 3,792 |  12,000,265 |  933 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/keccak-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 18,794 |  18,655,329 |  3,325 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/sha2_bench-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 10,049 |  14,793,960 |  1,446 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/regex-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 1,421 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/ecrecover-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 602 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/pairing-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 892 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/kitchen_sink-5112bdac6a69273022f9b22a82302be5573d5ed4.md) | 1,898 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5112bdac6a69273022f9b22a82302be5573d5ed4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26987218622)
