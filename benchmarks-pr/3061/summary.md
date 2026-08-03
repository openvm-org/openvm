| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-df776606a0f7ea92228024db902f43941c3f3362.md) | 478 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-df776606a0f7ea92228024db902f43941c3f3362.md) | 7,301 |  14,365,133 |  1,501 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-df776606a0f7ea92228024db902f43941c3f3362.md) | 4,163 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-df776606a0f7ea92228024db902f43941c3f3362.md) | 659 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-df776606a0f7ea92228024db902f43941c3f3362.md) | 230 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-df776606a0f7ea92228024db902f43941c3f3362.md) | 241 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-df776606a0f7ea92228024db902f43941c3f3362.md) | 2,050 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/df776606a0f7ea92228024db902f43941c3f3362

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30827565177)
