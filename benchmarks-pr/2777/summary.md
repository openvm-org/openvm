| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 1,857 |  4,000,051 |  443 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 13,916 |  14,365,133 |  2,214 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 8,150 |  11,167,961 |  910 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 1,556 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 606 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 738 |  592,827 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-5e2748cd4d4c6979f937f4153e784f635b0ce415.md) | 1,901 |  1,979,971 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5e2748cd4d4c6979f937f4153e784f635b0ce415

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26056113672)
