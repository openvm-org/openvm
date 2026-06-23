| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 1,020 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 16,230 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 8,197 |  11,167,961 |  1,006 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 1,184 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 435 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 591 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-36f30d984922c4aeb474585e39fadd93f77fc43e.md) | 3,835 |  1,979,971 |  847 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36f30d984922c4aeb474585e39fadd93f77fc43e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28046781337)
