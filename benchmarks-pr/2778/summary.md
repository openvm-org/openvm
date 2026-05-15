| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-1a396655b09449e89c47f25c187542e485641a21.md) | 1,417 |  4,000,051 |  439 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-1a396655b09449e89c47f25c187542e485641a21.md) | 13,236 |  14,365,133 |  2,188 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-1a396655b09449e89c47f25c187542e485641a21.md) | 8,883 |  11,167,961 |  1,384 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-1a396655b09449e89c47f25c187542e485641a21.md) | 1,334 |  4,090,656 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-1a396655b09449e89c47f25c187542e485641a21.md) | 472 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-1a396655b09449e89c47f25c187542e485641a21.md) | 599 |  592,827 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-1a396655b09449e89c47f25c187542e485641a21.md) | 1,786 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1a396655b09449e89c47f25c187542e485641a21

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25937055728)
