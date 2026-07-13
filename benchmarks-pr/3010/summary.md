| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/fibonacci-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 871 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/keccak-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 15,125 |  14,365,133 |  2,982 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/sha2_bench-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 8,213 |  11,167,961 |  1,027 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/regex-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 1,037 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/ecrecover-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 309 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/pairing-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 453 |  592,827 |  302 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3010/kitchen_sink-80607f2ca987e48d3dd133347647475c8d14cd76.md) | 3,707 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80607f2ca987e48d3dd133347647475c8d14cd76

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29272461108)
