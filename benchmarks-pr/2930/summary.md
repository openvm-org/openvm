| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/fibonacci-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 1,035 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/keccak-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 16,120 |  14,365,133 |  3,012 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/sha2_bench-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 8,345 |  11,167,961 |  1,019 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/regex-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 1,196 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/ecrecover-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 435 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/pairing-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 601 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2930/kitchen_sink-f32b41e98433d526fefca1dfa4e380a3589a3061.md) | 3,943 |  1,979,971 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f32b41e98433d526fefca1dfa4e380a3589a3061

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28125346316)
