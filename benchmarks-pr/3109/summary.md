| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 480 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 7,403 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 4,157 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 675 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 220 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 230 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-4dce656ddac72704371d48ab8da65c7718cd590a.md) | 2,038 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4dce656ddac72704371d48ab8da65c7718cd590a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31544444459)
