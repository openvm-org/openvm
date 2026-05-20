| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/fibonacci-0a09748bfeb5e02113976116247517a3c500b48a.md) | 3,735 |  12,000,265 | <span style='color: green'>(-3577 [-79.7%])</span> 909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/keccak-0a09748bfeb5e02113976116247517a3c500b48a.md) | 18,786 |  18,655,329 |  3,321 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/sha2_bench-0a09748bfeb5e02113976116247517a3c500b48a.md) | 10,252 |  14,793,960 |  1,461 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/regex-0a09748bfeb5e02113976116247517a3c500b48a.md) | 1,400 |  4,137,067 | <span style='color: green'>(-11644 [-97.1%])</span> 353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/ecrecover-0a09748bfeb5e02113976116247517a3c500b48a.md) | 600 |  123,583 | <span style='color: green'>(-5605 [-95.7%])</span> 251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/pairing-0a09748bfeb5e02113976116247517a3c500b48a.md) | 899 |  1,745,757 | <span style='color: green'>(-6114 [-95.8%])</span> 266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2798/kitchen_sink-0a09748bfeb5e02113976116247517a3c500b48a.md) | 1,903 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a09748bfeb5e02113976116247517a3c500b48a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26185983869)
