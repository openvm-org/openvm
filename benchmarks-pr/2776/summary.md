| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 3,807 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 19,007 |  18,655,329 |  3,345 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 9,006 |  14,793,960 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 1,444 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 638 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 907 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-93f0d1c59a4632e78c6d95aec67ab95034fe4f89.md) | 2,037 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/93f0d1c59a4632e78c6d95aec67ab95034fe4f89

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25783540495)
