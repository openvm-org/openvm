| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 1,408 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 13,315 |  14,365,133 |  2,201 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 9,096 |  11,167,961 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 1,334 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 468 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 589 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-0d37807f81ca10c7aa550cd241eac41abde64fd6.md) | 1,780 |  1,979,971 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0d37807f81ca10c7aa550cd241eac41abde64fd6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25926885834)
