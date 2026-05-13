| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 1,621 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 14,229 |  14,365,133 |  2,269 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 9,490 |  11,167,961 |  1,433 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 1,522 |  4,090,656 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 530 |  112,210 |  301 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 622 |  592,827 |  272 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.md) | 1,964 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25830117427)
