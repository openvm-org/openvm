| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 453 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 7,240 |  14,365,133 |  1,589 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 4,014 |  11,167,961 |  512 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 728 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 207 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 243 |  592,827 |  169 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-5ef177fa349546545aa2c0ff266aa39a4e3af2cf.md) | 2,167 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5ef177fa349546545aa2c0ff266aa39a4e3af2cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33151361198)
