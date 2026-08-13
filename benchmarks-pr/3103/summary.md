| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 458 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 7,386 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 4,190 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 673 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 193 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 236 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-5d89f29244acffdc6452e839ee3ff96789f61b54.md) | 2,048 |  1,979,971 |  528 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5d89f29244acffdc6452e839ee3ff96789f61b54

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31740197047)
