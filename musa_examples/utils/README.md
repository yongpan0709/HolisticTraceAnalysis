# musa_examples/utils - 计时工具模块

该模块提供了多种测量代码执行时间的方法，帮助快速定位性能瓶颈。

## 文件结构

```
musa_examples/utils/
├── __init__.py              # 模块初始化
├── timing.py                # 计时工具核心实现
├── example_timing_usage.py  # 使用示例
└── README.md                # 本文档
```

## 快速开始

### 方式一：装饰器（最简单）

```python
from musa_examples.utils import time_it

@time_it("function_name")
def my_function():
    # 你的代码
    pass

# 调用函数时会自动打印执行时间
my_function()
# 输出: [TIMING] function_name: 1.2345s (0.02min)
```

### 方式二：上下文管理器（推荐）

```python
from musa_examples.utils import get_timer, reset_timer

# 重置计时器（可选）
reset_timer()
timer = get_timer()

# 使用上下文管理器
with timer.measure("data_loading"):
    load_data()

with timer.measure("analysis"):
    analyze()

# 打印完整摘要
timer.print_summary()
```

### 方式三：快速计时器（不依赖全局）

```python
from musa_examples.utils import QuickTimer

# 上下文管理器方式
with QuickTimer("operation"):
    do_something()

# 手动控制方式
timer = QuickTimer("manual")
timer.start()
do_something()
elapsed = timer.stop()
```

## 在现有代码中的应用

### 修改 trace_etl.py

```python
from musa_examples.utils import get_timer, reset_timer

def main():
    # 重置计时器
    reset_timer()
    timer = get_timer()
    
    # 参数解析
    with timer.measure("argument_parsing"):
        parser = argparse.ArgumentParser(...)
        args = parser.parse_args()
    
    # Trace 加载
    with timer.measure("trace_loading"):
        trace_files = get_trace_files(args.trace_dir)
    
    # ETL 处理
    with timer.measure("etl_processing"):
        analyzer = DistributedMegatronTraceAnalysis(...)
        analyzer.pp_etl(output_dir, filter_out_funcs)
    
    # 打印完整计时报告
    print("\n" + timer.summary())
```

### 修改 distribute_trace_analysis.py

```python
from musa_examples.utils import time_it, get_timer

class DistributedMegatronTraceAnalysis:
    
    @time_it("initialization")
    def __init__(self, trace_dir: str, ...):
        timer = get_timer()
        
        with timer.measure("rank_generation"):
            self.expert_decoder_rank_generator = RankGenerator(...)
        
        with timer.measure("trace_file_discovery"):
            self.trace_files = get_trace_files(trace_dir)
    
    def pp_etl(self, output_dir: str, etl_func):
        timer = get_timer()
        
        with timer.measure("mpi_initialization"):
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        
        with timer.measure("etl_processing"):
            # ETL 处理逻辑
            pass
        
        # 在 root rank 打印计时摘要
        if rank == 0:
            print(timer.summary())
```

### 修改 parse_megatron.py

```python
from musa_examples.utils import time_it, reset_timer, get_timer

@time_it("main_function")
def main():
    reset_timer()
    timer = get_timer()
    
    with timer.measure("trace_analysis"):
        dist_megatron_analysis = DistributedMegatronTraceAnalysis(...)
    
    with timer.measure("analyze"):
        dist_megatron_analysis.analyze()
    
    # 打印摘要
    timer.print_summary()

if __name__ == '__main__':
    main()
```

## MPI 环境下的使用

```python
from mpi4py import MPI
from musa_examples.utils.timing import MPITimer

comm = MPI.COMM_WORLD

# 只在 root rank 打印计时信息
with MPITimer(comm, name="distributed_operation", root=0):
    # MPI 操作
    pass
```

## 输出示例

```
======================================================================
TIMING SUMMARY
======================================================================
trace_loading                                45.6789s (    0.76min)
etl_processing                              123.4567s (    2.06min)
mpi_communication                            12.3456s (    0.21min)
argument_parsing                              0.1234s (    0.00min)
----------------------------------------------------------------------
TOTAL                                       181.6046s (    3.03min)
======================================================================
```

## API 参考

### TimingTracker 类

| 方法 | 说明 |
|------|------|
| `start(name)` | 开始计时 |
| `stop(name)` | 停止计时，返回执行时间 |
| `measure(name, verbose=True)` | 上下文管理器 |
| `get_elapsed(name)` | 获取已完成的计时时间 |
| `clear()` | 清除所有记录 |
| `summary(sort_by_time=True)` | 生成摘要字符串 |
| `print_summary()` | 打印摘要 |
| `to_dict()` | 转换为字典 |

### 全局函数

| 函数 | 说明 |
|------|------|
| `get_timer()` | 获取全局计时器 |
| `reset_timer()` | 重置全局计时器 |
| `time_it(name)` | 装饰器 |
| `measure_time(name)` | 便捷上下文管理器 |

### 辅助类

| 类 | 说明 |
|-----|------|
| `QuickTimer` | 快速计时器（不依赖全局） |
| `MPITimer` | MPI 环境计时器 |

## 运行示例

```bash
cd musa_examples/utils
python example_timing_usage.py
```

## 最佳实践

1. **在脚本开始时调用 `reset_timer()`**：确保每次运行都是新的计时记录
2. **使用有意义的名称**：如 `trace_loading`、`data_processing` 等
3. **在关键函数使用装饰器**：`@time_it("function_name")`
4. **在代码块使用上下文管理器**：`with timer.measure("block_name")`
5. **在脚本结束时打印摘要**：`timer.print_summary()`
6. **MPI 环境只在 root 打印**：使用 `MPITimer` 或检查 rank

## 性能影响

计时工具使用 `time.perf_counter()`，精度高且开销极小（< 1μs），不会影响实际性能分析。