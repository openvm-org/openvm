| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 408 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 8,632 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 4,204 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 573 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 218 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 296 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-6ba562f3a1d9412432014bb72e69bb9199461b8c.md) | 1,922 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6ba562f3a1d9412432014bb72e69bb9199461b8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29654115350)
