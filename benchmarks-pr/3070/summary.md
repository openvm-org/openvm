| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 461 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 7,245 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 4,729 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 652 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 226 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 299 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-33ada7b1d3e3efa363d341e1e97b45a43e15d3ba.md) | 2,643 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33ada7b1d3e3efa363d341e1e97b45a43e15d3ba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30433148182)
