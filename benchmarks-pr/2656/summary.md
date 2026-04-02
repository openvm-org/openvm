| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/fibonacci-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 3,834 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/keccak-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 15,678 |  1,235,218 |  2,197 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/regex-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 1,414 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/ecrecover-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 637 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/pairing-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 926 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/kitchen_sink-d6173ef6001e065ee1be50b4b426b3989d227965.md) | 2,385 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d6173ef6001e065ee1be50b4b426b3989d227965

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23919704282)
