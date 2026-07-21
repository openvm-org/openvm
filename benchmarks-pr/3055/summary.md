| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 407 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 8,669 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 4,230 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 573 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 219 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 297 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-e0a6c78f4569af6a16c8932a627a46c392bb07a9.md) | 1,921 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e0a6c78f4569af6a16c8932a627a46c392bb07a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29860264983)
