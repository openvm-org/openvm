| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 4,132 |  12,000,265 |  1,353 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 19,259 |  1,235,218 |  3,388 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 1,613 |  4,136,694 |  533 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 648 |  122,348 |  343 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 1,054 |  1,745,757 |  346 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-fc9ed7b2faddb853be78aef39c88df9941d256c3.md) | 3,278 |  154,763 |  723 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fc9ed7b2faddb853be78aef39c88df9941d256c3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23567782202)
