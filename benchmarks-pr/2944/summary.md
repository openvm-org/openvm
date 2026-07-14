| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 477 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 8,890 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 3,933 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 514 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 219 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 275 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-28cda984bb3b67f481ab80bcfd8ab65744b46171.md) | 1,921 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/28cda984bb3b67f481ab80bcfd8ab65744b46171

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29373550715)
