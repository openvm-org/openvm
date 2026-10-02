| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/fibonacci-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 487 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/keccak-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 7,778 |  14,365,133 |  1,557 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/sha2_bench-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 4,271 |  11,167,961 |  536 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/regex-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 756 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/ecrecover-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 212 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/pairing-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 255 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/kitchen_sink-bd578997c705d20f03d72248bc4efe5b24f40666.md) | 2,231 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bd578997c705d20f03d72248bc4efe5b24f40666

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/37041476225)
