| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/fibonacci-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 1,042 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/keccak-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 15,608 |  14,365,133 |  3,007 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/sha2_bench-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 8,223 |  11,167,961 |  1,011 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/regex-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 1,171 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/ecrecover-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 436 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/pairing-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 599 |  592,827 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/kitchen_sink-f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026.md) | 3,884 |  1,979,971 |  866 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f0e5fdd6ec20258a24d7e9c7e53f8cbf5ac0b026

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28257339279)
