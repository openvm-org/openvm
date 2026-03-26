| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/fibonacci-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 3,833 |  12,000,265 |  940 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/keccak-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 15,683 |  1,235,218 |  2,195 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/regex-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 1,433 |  4,136,694 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/ecrecover-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 648 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/pairing-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 913 |  1,745,757 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/kitchen_sink-a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb.md) | 2,374 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8d3731ddc5d8e193925b4aa5cc6ce98971a7adb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23623020747)
