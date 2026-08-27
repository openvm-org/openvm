| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-0f29970767ad670f63d561f14a8f44e311da8233.md) | 447 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-0f29970767ad670f63d561f14a8f44e311da8233.md) | 7,214 |  14,365,133 |  1,575 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-0f29970767ad670f63d561f14a8f44e311da8233.md) | 4,087 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-0f29970767ad670f63d561f14a8f44e311da8233.md) | 729 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-0f29970767ad670f63d561f14a8f44e311da8233.md) | 206 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-0f29970767ad670f63d561f14a8f44e311da8233.md) | 237 |  592,827 |  167 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-0f29970767ad670f63d561f14a8f44e311da8233.md) | 2,141 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0f29970767ad670f63d561f14a8f44e311da8233

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33110630599)
