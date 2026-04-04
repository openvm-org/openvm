| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/fibonacci-cd93299158daefc4dfcea4924f47186029e84e63.md) | 3,820 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/keccak-cd93299158daefc4dfcea4924f47186029e84e63.md) | 18,868 |  18,655,329 |  3,385 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/regex-cd93299158daefc4dfcea4924f47186029e84e63.md) | 1,409 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/ecrecover-cd93299158daefc4dfcea4924f47186029e84e63.md) | 647 |  123,583 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/pairing-cd93299158daefc4dfcea4924f47186029e84e63.md) | 923 |  1,745,757 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/kitchen_sink-cd93299158daefc4dfcea4924f47186029e84e63.md) | 2,287 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cd93299158daefc4dfcea4924f47186029e84e63

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23968547294)
