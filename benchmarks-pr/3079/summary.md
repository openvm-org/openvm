| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/fibonacci-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 468 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/keccak-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 7,255 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/sha2_bench-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 4,759 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/regex-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 671 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/ecrecover-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 229 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/pairing-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 312 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/kitchen_sink-0312679c4d3710c12d575fe33e7966512cac6db6.md) | 2,676 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0312679c4d3710c12d575fe33e7966512cac6db6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30317407470)
