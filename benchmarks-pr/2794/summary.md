| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 1,611 |  4,000,051 |  456 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 13,929 |  14,365,133 |  2,360 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 9,365 |  11,167,961 |  1,424 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 1,614 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 489 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 610 |  592,827 |  254 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-2bbbed568140bd72b127e1e996af2c54385e8210.md) | 1,820 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2bbbed568140bd72b127e1e996af2c54385e8210

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26829675775)
