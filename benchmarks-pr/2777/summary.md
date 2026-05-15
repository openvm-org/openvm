| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 1,828 |  4,000,051 |  438 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 13,829 |  14,365,133 |  2,193 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 8,072 |  11,167,961 |  896 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 1,514 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 610 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 740 |  592,827 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-64815b88cc8ae5af27517a12f8539ebcdfdc73fe.md) | 1,882 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64815b88cc8ae5af27517a12f8539ebcdfdc73fe

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25937515618)
