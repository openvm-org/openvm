| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 1,578 |  4,000,051 |  441 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 13,606 |  14,365,133 |  2,330 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 9,225 |  11,167,961 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 1,493 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 470 |  112,210 |  261 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 595 |  592,827 |  254 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-8658dfa6e706c8887c3cd0d14c3df082f5da7953.md) | 1,867 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8658dfa6e706c8887c3cd0d14c3df082f5da7953

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887659271)
