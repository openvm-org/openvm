| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/fibonacci-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 473 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/keccak-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 7,322 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/sha2_bench-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 4,161 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/regex-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 658 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/ecrecover-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 223 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/pairing-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 230 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/kitchen_sink-0ec64a6b2c00c85461c7b7697180ef294d9b7f09.md) | 2,027 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0ec64a6b2c00c85461c7b7697180ef294d9b7f09

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31027649381)
