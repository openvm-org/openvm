| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 1,927 |  4,000,051 |  538 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 13,544 |  14,365,133 |  2,229 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 9,398 |  11,167,961 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 1,598 |  4,090,656 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 639 |  112,210 |  290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 753 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 2,042 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9a37c324d636150b631dab01cedfe4f413b93f41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25802526739)
