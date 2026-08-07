| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-1846726d83317399d96eb47954df974447417392.md) | 437 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-1846726d83317399d96eb47954df974447417392.md) | 7,095 |  14,365,133 |  1,577 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-1846726d83317399d96eb47954df974447417392.md) | 4,104 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-1846726d83317399d96eb47954df974447417392.md) | 711 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-1846726d83317399d96eb47954df974447417392.md) | 204 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-1846726d83317399d96eb47954df974447417392.md) | 236 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-1846726d83317399d96eb47954df974447417392.md) | 2,170 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1846726d83317399d96eb47954df974447417392

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31224278300)
