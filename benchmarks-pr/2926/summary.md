| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/fibonacci-99060bf0ecd740a70193f005a3567028675abd5e.md) | 1,027 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/keccak-99060bf0ecd740a70193f005a3567028675abd5e.md) | 16,509 |  14,365,133 |  3,055 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/sha2_bench-99060bf0ecd740a70193f005a3567028675abd5e.md) | 8,127 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/regex-99060bf0ecd740a70193f005a3567028675abd5e.md) | 1,230 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/ecrecover-99060bf0ecd740a70193f005a3567028675abd5e.md) | 433 |  112,210 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/pairing-99060bf0ecd740a70193f005a3567028675abd5e.md) | 594 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/kitchen_sink-99060bf0ecd740a70193f005a3567028675abd5e.md) | 3,877 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/99060bf0ecd740a70193f005a3567028675abd5e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28064281766)
