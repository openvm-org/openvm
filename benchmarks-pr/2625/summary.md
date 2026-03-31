| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 3,865 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 15,665 |  1,235,218 |  2,162 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 1,412 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 641 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 932 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-0b423cbf5135a83cd19bf77c752dcf95d3aae328.md) | 2,385 |  154,763 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0b423cbf5135a83cd19bf77c752dcf95d3aae328

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23817241717)
