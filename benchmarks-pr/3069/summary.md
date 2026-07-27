| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/fibonacci-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 460 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/keccak-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 7,253 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/sha2_bench-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 4,723 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/regex-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 646 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/ecrecover-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 234 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/pairing-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 306 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/kitchen_sink-55ca06f1b81aae831339d4b9182eec145ada358f.md) | 2,631 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/55ca06f1b81aae831339d4b9182eec145ada358f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30286521967)
