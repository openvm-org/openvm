| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 478 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 7,427 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 4,163 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 654 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 240 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-209bec08102289fd03fa2731870fbbc4bed5a289.md) | 2,046 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/209bec08102289fd03fa2731870fbbc4bed5a289

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30847906651)
