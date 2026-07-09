| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 864 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 15,251 |  14,365,133 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 7,525 |  11,167,961 |  985 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 875 |  4,090,656 |  298 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 298 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 406 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-eed6502cc117f59c8e74239c55f982bb99f0c20d.md) | 3,647 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eed6502cc117f59c8e74239c55f982bb99f0c20d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29055295380)
