| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 3,784 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 18,386 |  18,655,329 |  3,290 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 1,426 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 647 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 919 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-506481c41e19f7e4cd06ef39f25482dacd450465.md) | 2,291 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/506481c41e19f7e4cd06ef39f25482dacd450465

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24037080317)
