| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-02c02e34b705393b2405731ef2f07ca00b351289.md) | 479 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-02c02e34b705393b2405731ef2f07ca00b351289.md) | 10,267 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-02c02e34b705393b2405731ef2f07ca00b351289.md) | 4,656 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-02c02e34b705393b2405731ef2f07ca00b351289.md) | 679 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-02c02e34b705393b2405731ef2f07ca00b351289.md) | 231 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-02c02e34b705393b2405731ef2f07ca00b351289.md) | 277 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-02c02e34b705393b2405731ef2f07ca00b351289.md) | 2,373 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/02c02e34b705393b2405731ef2f07ca00b351289

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30138636411)
