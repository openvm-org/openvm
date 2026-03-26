| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 3,812 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 15,693 |  1,235,218 |  2,171 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 1,431 |  4,136,694 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 634 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 922 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-bb392351bf3e65d3cadd8660ee1b598d7bd08089.md) | 2,380 |  154,763 |  402 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bb392351bf3e65d3cadd8660ee1b598d7bd08089

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23616825370)
