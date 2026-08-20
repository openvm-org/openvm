| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 460 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 7,309 |  14,365,133 |  1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 4,053 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 706 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 202 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 235 |  592,827 |  167 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-19b412a9e0d11b1522128705a0dd978c10af6afe.md) | 2,172 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19b412a9e0d11b1522128705a0dd978c10af6afe

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32414207279)
