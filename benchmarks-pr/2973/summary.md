| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 438 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 8,575 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 4,143 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 501 |  4,090,656 |  195 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 224 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 274 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-db4687eb340c6950da8ca95687c8c4ea473a53c3.md) | 2,017 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db4687eb340c6950da8ca95687c8c4ea473a53c3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29398471433)
