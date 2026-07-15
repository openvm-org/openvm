| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/fibonacci-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 411 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/keccak-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 8,383 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/sha2_bench-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 3,951 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/regex-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 580 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/ecrecover-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 218 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/pairing-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 260 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/kitchen_sink-66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba.md) | 1,890 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/66b9bb18fe9208bba4ce8acf5dae64c6526ad4ba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29398434974)
