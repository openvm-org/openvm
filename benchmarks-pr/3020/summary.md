| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 476 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 7,270 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 4,775 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 694 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 231 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 278 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-73e76c5f3f3b92b7f9ff7dc90971cde0b1981885.md) | 2,742 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/73e76c5f3f3b92b7f9ff7dc90971cde0b1981885

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29752393567)
