| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 451 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 7,218 |  14,365,133 |  1,597 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 4,084 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 741 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 206 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 241 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-a19d3561108611d7f626aa170ca87632ee7f66e8.md) | 2,154 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a19d3561108611d7f626aa170ca87632ee7f66e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32513341685)
