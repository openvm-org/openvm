| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/fibonacci-2165e94b00571f92136aa55edaba576d25e88c37.md) | 479 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/keccak-2165e94b00571f92136aa55edaba576d25e88c37.md) | 7,479 |  14,365,133 |  1,558 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/sha2_bench-2165e94b00571f92136aa55edaba576d25e88c37.md) | 4,125 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/regex-2165e94b00571f92136aa55edaba576d25e88c37.md) | 656 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/ecrecover-2165e94b00571f92136aa55edaba576d25e88c37.md) | 229 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/pairing-2165e94b00571f92136aa55edaba576d25e88c37.md) | 234 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3093/kitchen_sink-2165e94b00571f92136aa55edaba576d25e88c37.md) | 2,041 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2165e94b00571f92136aa55edaba576d25e88c37

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30840150524)
