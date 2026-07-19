| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 406 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 8,646 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 4,193 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 570 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 221 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 281 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-a2b4651d646f3fe46fb23f384a7427813cc942ac.md) | 1,910 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2b4651d646f3fe46fb23f384a7427813cc942ac

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29698347645)
