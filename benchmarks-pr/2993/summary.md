| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 405 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 8,715 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 4,177 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 576 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 227 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 295 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395.md) | 1,907 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/75ea5489ef6811ad9f57b11bb7ab1d44bf0e2395

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29535760267)
