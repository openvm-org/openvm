| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-d457878355df5a7068acd88537818f40dae614bf.md) | 1,555 |  4,000,051 |  438 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-d457878355df5a7068acd88537818f40dae614bf.md) | 13,965 |  14,365,133 |  2,378 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-d457878355df5a7068acd88537818f40dae614bf.md) | 9,193 |  11,167,961 |  1,426 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-d457878355df5a7068acd88537818f40dae614bf.md) | 1,586 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-d457878355df5a7068acd88537818f40dae614bf.md) | 488 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-d457878355df5a7068acd88537818f40dae614bf.md) | 596 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-d457878355df5a7068acd88537818f40dae614bf.md) | 1,991 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d457878355df5a7068acd88537818f40dae614bf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26946137069)
