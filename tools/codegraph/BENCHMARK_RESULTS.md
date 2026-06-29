# Benchmark Results: CodeGraph vs PyCG (ICSE 2021)

## Benchmark Source

**PyCG Micro-Benchmark Suite**  
- Paper: "PyCG: Practical Call Graph Generation in Python" (ICSE 2021)  
- Repository: [github.com/vitsalis/PyCG](https://github.com/vitsalis/PyCG)  
- The most cited Python call graph benchmark in the static analysis community.
- 119 micro-benchmark test cases across 18 categories

## Results Summary

| Metric | CodeGraph | PyCG (Baseline) |
|--------|-----------|-----------------|
| **Precision** | **76.6%** | 99.2% |
| **Recall** | **37.1%** | 69.9% |
| **F1** | **50.0%** | 82.2% |
| Tests Run | 119 | 119 |
| Crashed | 0 | 0 |

## Results by Category

| Category | P | R | F1 | TP | FP | FN|
|----------|------|------|------|-----|-----|------|
| args | 1.00 | 0.43 | 0.60 | 6 | 0 | 8 |
| assignments | 1.00 | 0.00 | 0.00 | 0 | 0 | 15 |
| builtins | 1.00 | 0.10 | 0.18 | 1 | 0 | 9 |
| **classes** | **0.72** | **0.65** | **0.69** | 34 | 13 | 18 |
| decorators | 0.89 | 0.36 | 0.52 | 8 | 1 | 14 |
| dicts | 1.00 | 0.26 | 0.42 | 5 | 0 | 14 |
| direct_calls | 1.00 | 0.30 | 0.46 | 3 | 0 | 7 |
| dynamic | 1.00 | 0.00 | 0.00 | 0 | 0 | 2 |
| exceptions | 1.00 | 0.00 | 0.00 | 0 | 0 | 3 |
| external | 0.67 | 0.18 | 0.29 | 2 | 1 | 9 |
| functions | 1.00 | 0.25 | 0.40 | 1 | 0 | 3 |
| generators | 1.00 | 0.39 | 0.56 | 7 | 0 | 11 |
| imports | 0.33 | 0.36 | 0.34 | 5 | 10 | 9 |
| kwargs | 1.00 | 0.20 | 0.33 | 2 | 0 | 8 |
| lambdas | 1.00 | 0.36 | 0.53 | 5 | 0 | 9 |
| lists | 1.00 | 0.31 | 0.48 | 5 | 0 | 11 |
| mro | 0.55 | 0.33 | 0.41 | 6 | 5 | 12 |
| **returns** | **1.00** | **0.67** | **0.80** | 8 | 0 | 4 |

## Strengths

CodeGraph performs well on categories that involve direct structural analysis:

- **Classes (F1=0.69)**: Class instantiation, method calls, `__init__` resolution
- **Returns (F1=0.80)**: Return value tracking and function composition
- **Args (F1=0.60)**: Argument passing and function call patterns
- **Generators (F1=0.56)**: Iterator and yield-based call patterns
- **Lambdas (F1=0.53)**: Lambda function call resolution

## Known Weaknesses

### Assignments (R=0%, 15 FN)
```python
a = func
a()  # Cannot resolve: requires data-flow analysis
```
CodeGraph does not track variable assignments to resolve indirect calls. This requires inter-procedural data-flow analysis — a fundamentally different (and much more expensive) technique.

### Dynamic (R=0%, 2 FN)
```python
eval("func()")  # Cannot resolve: requires runtime execution
```

### Builtins (R=10%, 9 FN)
Built-in function calls (e.g., `map(func, list)`) where the function is passed as an argument.

## Design Differences: CodeGraph vs PyCG

CodeGraph and PyCG solve **different problems**:

| Aspect | PyCG | CodeGraph |
|--------|------|-----------|
| **Goal** | Complete call graph | Architectural reasoning |
| **Analysis Type** | Inter-procedural data flow | AST + graph traversal |
| **Assignment Tracking** | ✅ Yes | ❌ No |
| **Dynamic Dispatch** | Partial | ❌ No |
| **Multi-hop Impact** | Not designed for this | ✅ Up to 12+ hops |
| **Community Detection** | No | ✅ Leiden Algorithm |
| **Indexing Speed** | Seconds | **<1s** |

## Optimization History

| Round | Change | Precision | Recall | F1 |
|-------|--------|-----------|--------|-----|
| 0 | Baseline | 75.0% | 2.3% | 4.4% |
| 1 | Module-scope call tracking | 52.2% | 17.8% | 26.6% |
| 2 | Class → `__init__` resolution | 76.3% | 33.0% | 46.0% |
| 3 | `self.method()` scope fix | **76.6%** | **37.1%** | **50.0%** |
