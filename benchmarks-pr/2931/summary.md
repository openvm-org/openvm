| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-d6dea47f280f8d778297a8e0972787525b396b30.md) | 1,025 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-d6dea47f280f8d778297a8e0972787525b396b30.md) | 15,652 |  14,365,133 |  3,002 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-d6dea47f280f8d778297a8e0972787525b396b30.md) | 8,114 |  11,167,961 |  1,003 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-d6dea47f280f8d778297a8e0972787525b396b30.md) | 1,167 |  4,090,656 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-d6dea47f280f8d778297a8e0972787525b396b30.md) | 436 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-d6dea47f280f8d778297a8e0972787525b396b30.md) | 584 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-d6dea47f280f8d778297a8e0972787525b396b30.md) | 3,880 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d6dea47f280f8d778297a8e0972787525b396b30

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28320799190)
