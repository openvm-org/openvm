| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-757717f358b696870fae0a0b1294cc344a1eff35.md) | 3,779 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-757717f358b696870fae0a0b1294cc344a1eff35.md) | 18,621 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-757717f358b696870fae0a0b1294cc344a1eff35.md) | 10,186 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-757717f358b696870fae0a0b1294cc344a1eff35.md) | 1,392 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-757717f358b696870fae0a0b1294cc344a1eff35.md) | 601 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-757717f358b696870fae0a0b1294cc344a1eff35.md) | 887 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-757717f358b696870fae0a0b1294cc344a1eff35.md) | 1,899 |  2,579,903 |  412 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-757717f358b696870fae0a0b1294cc344a1eff35.md) | 1,782 |  12,000,265 |  409 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-757717f358b696870fae0a0b1294cc344a1eff35.md) | 815 |  4,137,067 |  172 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-757717f358b696870fae0a0b1294cc344a1eff35.md) | 514 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-757717f358b696870fae0a0b1294cc344a1eff35.md) | 627 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-757717f358b696870fae0a0b1294cc344a1eff35.md) | 2,027 |  2,579,903 |  400 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/757717f358b696870fae0a0b1294cc344a1eff35

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26662142784)
