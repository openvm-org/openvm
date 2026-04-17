| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 3,867 |  12,000,265 |  962 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 18,783 |  18,655,329 |  3,356 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 8,968 |  14,793,960 |  1,390 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 1,418 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 648 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 905 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 2,098 |  2,579,903 |  436 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 1,858 |  12,000,265 |  455 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 856 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 555 |  123,583 |  153 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 656 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-c9db07129debb033f62eb92bf88da1f2ab38fd4b.md) | 2,225 |  2,579,903 |  425 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c9db07129debb033f62eb92bf88da1f2ab38fd4b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24582528431)
