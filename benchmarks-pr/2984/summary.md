| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-f76ea467d9918370f12870e767dc976d868d316b.md) | 475 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-f76ea467d9918370f12870e767dc976d868d316b.md) | 7,300 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-f76ea467d9918370f12870e767dc976d868d316b.md) | 4,707 |  11,167,961 |  538 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-f76ea467d9918370f12870e767dc976d868d316b.md) | 666 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-f76ea467d9918370f12870e767dc976d868d316b.md) | 224 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-f76ea467d9918370f12870e767dc976d868d316b.md) | 269 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-f76ea467d9918370f12870e767dc976d868d316b.md) | 2,733 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f76ea467d9918370f12870e767dc976d868d316b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30058215173)
