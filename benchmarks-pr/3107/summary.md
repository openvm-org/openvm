| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/fibonacci-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 475 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/keccak-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 7,342 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/sha2_bench-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 4,135 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/regex-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 656 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/ecrecover-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 223 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/pairing-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 229 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/kitchen_sink-859ade7cf00da4bae52ed6dd35a6cb722c136c7c.md) | 2,038 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/859ade7cf00da4bae52ed6dd35a6cb722c136c7c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31034228202)
