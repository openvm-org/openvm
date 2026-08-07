| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 886 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 8,598 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 4,210 |  11,167,961 |  515 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 739 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 499 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 474 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-22233e8d04e005d72a73cac1a52c8550d4442da7.md) | 2,361 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/22233e8d04e005d72a73cac1a52c8550d4442da7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31202349265)
