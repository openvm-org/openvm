| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 554 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 7,481 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 4,545 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 609 |  4,090,656 |  193 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 253 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-8b593c75b56dec1cf7ee5380ba3e3b04d81850f7.md) | 2,681 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8b593c75b56dec1cf7ee5380ba3e3b04d81850f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29395067486)
