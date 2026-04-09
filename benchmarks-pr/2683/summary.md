| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 3,812 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 18,691 |  18,655,329 |  3,353 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 1,423 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 658 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 899 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-cc9c4e0a91b31edf9694f209a60baad763732c8b.md) | 2,151 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cc9c4e0a91b31edf9694f209a60baad763732c8b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24205855304)
