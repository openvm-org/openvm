| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/fibonacci-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 1,023 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/keccak-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 16,280 |  14,365,133 |  3,010 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/sha2_bench-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 8,263 |  11,167,961 |  1,003 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/regex-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 1,229 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/ecrecover-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 439 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/pairing-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 603 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/kitchen_sink-8c00127cc8cc8ba4845a4da05a3011885020dbeb.md) | 3,849 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8c00127cc8cc8ba4845a4da05a3011885020dbeb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27974188673)
