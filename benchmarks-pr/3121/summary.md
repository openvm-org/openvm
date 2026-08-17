| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/fibonacci-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 466 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/keccak-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 7,507 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/sha2_bench-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 4,166 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/regex-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 658 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/ecrecover-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 199 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/pairing-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 233 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3121/kitchen_sink-f6f015ae7eaea87b3fabf8c1ccc9a556509ea040.md) | 2,045 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f6f015ae7eaea87b3fabf8c1ccc9a556509ea040

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32044927655)
