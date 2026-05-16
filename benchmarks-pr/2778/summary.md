| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 1,411 |  4,000,051 |  432 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 13,219 |  14,365,133 |  2,181 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 8,995 |  11,167,961 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 1,334 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 472 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 592 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-7fc99506c0e4d0492cbc4b0a6dfd399266db0479.md) | 2,209 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7fc99506c0e4d0492cbc4b0a6dfd399266db0479

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25967789566)
