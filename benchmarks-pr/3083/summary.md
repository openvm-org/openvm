| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 459 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 7,222 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 4,731 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 652 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 231 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 288 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-409a26fa06bb81c36ba0933c001fca94a93921d6.md) | 2,629 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/409a26fa06bb81c36ba0933c001fca94a93921d6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30569355820)
