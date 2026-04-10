| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 3,826 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 18,555 |  18,655,329 |  3,305 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 1,425 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 649 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-a7c060c60e4e2a910610da6175bf1bc5b542873c.md) | 2,153 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a7c060c60e4e2a910610da6175bf1bc5b542873c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24262394994)
