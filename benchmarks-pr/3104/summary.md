| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 453 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 7,254 |  14,365,133 |  1,588 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 4,177 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 725 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 204 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 237 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-1f9a3d66fc89df95423c23e09b30db7f11c77d36.md) | 2,174 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1f9a3d66fc89df95423c23e09b30db7f11c77d36

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32400630019)
