| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 469 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 7,634 |  14,365,133 |  1,629 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 4,276 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 739 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 208 |  112,210 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 255 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-a7ca43500f2e20e3fabfc6ee6c2c496869c5a358.md) | 2,241 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a7ca43500f2e20e3fabfc6ee6c2c496869c5a358

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34603158727)
