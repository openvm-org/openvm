| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-3537289f0983be1bf27a114465fed9579b7aa933.md) | 1,902 |  4,000,051 |  539 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-3537289f0983be1bf27a114465fed9579b7aa933.md) | 13,837 |  14,365,133 |  2,273 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-3537289f0983be1bf27a114465fed9579b7aa933.md) | 9,355 |  11,167,961 |  1,403 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-3537289f0983be1bf27a114465fed9579b7aa933.md) | 1,599 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-3537289f0983be1bf27a114465fed9579b7aa933.md) | 639 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-3537289f0983be1bf27a114465fed9579b7aa933.md) | 759 |  592,827 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-3537289f0983be1bf27a114465fed9579b7aa933.md) | 2,035 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3537289f0983be1bf27a114465fed9579b7aa933

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25865655106)
