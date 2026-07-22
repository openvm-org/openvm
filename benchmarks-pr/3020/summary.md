| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 482 |  4,000,051 |  245 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 7,342 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 4,762 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 690 |  4,090,656 |  223 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 276 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-44ded8ec25a72248a36e2135d89abce20f0b41e7.md) | 2,757 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/44ded8ec25a72248a36e2135d89abce20f0b41e7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29959895972)
