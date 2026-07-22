| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/fibonacci-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 469 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/keccak-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 7,269 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/sha2_bench-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 4,741 |  11,167,961 |  539 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/regex-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 669 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/ecrecover-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 230 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/pairing-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 314 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/kitchen_sink-c32a010b9ce816f1e7414ba0df5fc5794658d28f.md) | 2,659 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c32a010b9ce816f1e7414ba0df5fc5794658d28f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29937405867)
