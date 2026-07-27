| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 495 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 7,396 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 4,166 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 690 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 234 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 241 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-4584e73687fc4e3b023860bd446be9e0150a0550.md) | 2,053 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4584e73687fc4e3b023860bd446be9e0150a0550

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30304405962)
