| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/fibonacci-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 5,831 |  4,000,051 | <span style='color: green'>(-10190 [-95.0%])</span> 534 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/keccak-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 30,319 |  14,365,133 |  2,214 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/sha2_bench-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 22,630 |  11,167,961 |  1,410 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/regex-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 6,321 |  4,090,656 | <span style='color: green'>(-27326 [-98.6%])</span> 379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/ecrecover-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 648 |  112,210 | <span style='color: green'>(-10567 [-97.3%])</span> 290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/pairing-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 1,336 |  592,827 | <span style='color: green'>(-13866 [-98.0%])</span> 283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/kitchen_sink-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 4,553 |  1,979,971 |  430 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/fibonacci_e2e-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 5,515 |  4,000,051 |  251 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/regex_e2e-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 5,890 |  4,090,656 |  196 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/ecrecover_e2e-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 634 |  112,210 |  159 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/pairing_e2e-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 1,252 |  592,827 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-dispatch/refs/heads/develop-v2.1.0-rv64/kitchen_sink_e2e-9a37c324d636150b631dab01cedfe4f413b93f41.md) | 4,574 |  1,979,971 |  421 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9a37c324d636150b631dab01cedfe4f413b93f41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25825859790)
