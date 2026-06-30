| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/fibonacci-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 1,040 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/keccak-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 15,834 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/sha2_bench-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 8,298 |  11,167,961 |  1,022 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/regex-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 1,163 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/ecrecover-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 430 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/pairing-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 576 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/kitchen_sink-a1a5755c1a0cfade65f4d4f1131e65882828deab.md) | 3,858 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a1a5755c1a0cfade65f4d4f1131e65882828deab

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28477240422)
