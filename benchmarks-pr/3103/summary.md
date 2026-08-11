| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-427e15d75931197eabae49fd57f521d107ce907f.md) | 465 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-427e15d75931197eabae49fd57f521d107ce907f.md) | 7,399 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-427e15d75931197eabae49fd57f521d107ce907f.md) | 4,125 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-427e15d75931197eabae49fd57f521d107ce907f.md) | 660 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-427e15d75931197eabae49fd57f521d107ce907f.md) | 223 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-427e15d75931197eabae49fd57f521d107ce907f.md) | 230 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-427e15d75931197eabae49fd57f521d107ce907f.md) | 2,038 |  1,979,971 |  530 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/427e15d75931197eabae49fd57f521d107ce907f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31538277995)
