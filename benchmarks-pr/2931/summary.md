| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 1,022 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 16,076 |  14,365,133 |  3,006 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 8,085 |  11,167,961 |  990 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 1,182 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 438 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 583 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-fded558eb2aafa1bcc24bd7e9443b3477814095c.md) | 3,891 |  1,979,971 |  868 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fded558eb2aafa1bcc24bd7e9443b3477814095c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28172410266)
