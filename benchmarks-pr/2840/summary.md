| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-c279370741a9fc28af063675e39abff95ace2335.md) | 1,420 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-c279370741a9fc28af063675e39abff95ace2335.md) | 14,230 |  14,365,133 |  2,180 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-c279370741a9fc28af063675e39abff95ace2335.md) | 9,142 |  11,167,961 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-c279370741a9fc28af063675e39abff95ace2335.md) | 1,546 |  4,090,656 |  364 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-c279370741a9fc28af063675e39abff95ace2335.md) | 447 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-c279370741a9fc28af063675e39abff95ace2335.md) | 587 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-c279370741a9fc28af063675e39abff95ace2335.md) | 2,182 |  1,979,971 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c279370741a9fc28af063675e39abff95ace2335

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26913597951)
