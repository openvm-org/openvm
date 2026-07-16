| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 406 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 8,464 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 4,107 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 496 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 224 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 269 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae.md) | 1,883 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6ac4f2a3755bf2d4c0da259d2472aacd8cfc56ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29496606246)
