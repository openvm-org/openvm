| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/fibonacci-163416ed62b1c541addbef41cf0774934113bbe5.md) | 3,740 |  12,000,265 |  910 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/keccak-163416ed62b1c541addbef41cf0774934113bbe5.md) | 18,668 |  18,655,329 |  3,301 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/sha2_bench-163416ed62b1c541addbef41cf0774934113bbe5.md) | 10,170 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/regex-163416ed62b1c541addbef41cf0774934113bbe5.md) | 1,402 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/ecrecover-163416ed62b1c541addbef41cf0774934113bbe5.md) | 601 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/pairing-163416ed62b1c541addbef41cf0774934113bbe5.md) | 897 |  1,745,757 |  268 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/kitchen_sink-163416ed62b1c541addbef41cf0774934113bbe5.md) | 1,887 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/163416ed62b1c541addbef41cf0774934113bbe5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25930994027)
