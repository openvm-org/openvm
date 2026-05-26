| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/fibonacci-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 1,862 |  4,000,051 |  513 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/keccak-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 13,762 |  14,365,133 |  2,239 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/sha2_bench-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 9,443 |  11,167,961 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/regex-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 1,570 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/ecrecover-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 605 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/pairing-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 739 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/kitchen_sink-74e152389086908a262a1759f1a70e3b3f0f4bf1.md) | 1,865 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/74e152389086908a262a1759f1a70e3b3f0f4bf1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26478440355)
