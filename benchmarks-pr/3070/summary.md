| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 458 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 7,257 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 4,739 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 651 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 229 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 304 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-6e274c37426b752319bd4c76ebd466681df6de3b.md) | 2,636 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6e274c37426b752319bd4c76ebd466681df6de3b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30435801907)
