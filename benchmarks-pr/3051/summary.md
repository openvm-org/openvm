| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/fibonacci-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 412 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/keccak-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 8,723 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/sha2_bench-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 4,239 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/regex-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 573 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/ecrecover-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 222 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/pairing-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 282 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/kitchen_sink-42fa94ec2151bbad92362597b21af696ac526ddc.md) | 1,942 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/42fa94ec2151bbad92362597b21af696ac526ddc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29777098376)
