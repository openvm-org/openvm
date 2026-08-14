| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 463 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 7,377 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 4,191 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 666 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 196 |  112,210 |  201 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 237 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-a01e22fc072009e3076e3397fe5c31020b65f750.md) | 2,033 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a01e22fc072009e3076e3397fe5c31020b65f750

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31835117474)
