# Limitations & Known Error Cases

CodeGraph uses **static AST analysis** to build its knowledge graph. This is its greatest strength (deterministic, zero-cost) and also the source of its fundamental limitations.

This document exists because **acknowledging limitations is what separates research-grade work from marketing**.

---

## 1. Dynamic Dispatch

Python is a dynamic language. The following patterns are **invisible** to static analysis:

### `getattr(obj, name)()`
```python
# Static analysis sees: a function call to getattr
# It CANNOT resolve what 'name' evaluates to at runtime
handler = getattr(self, request.method.lower())
handler()  # ← callee is unknown
```
**Impact**: ~5-10% of calls in typical Flask apps use this pattern.  
**Mitigation**: Constant-string `getattr` analysis (when the second argument is a string literal) could resolve ~40% of these cases.

### `self.view_functions[endpoint]()`
```python
# The view function is stored in a dict, keyed by a runtime string
rv = self.view_functions[rule.endpoint](**req.view_args)
```
**Impact**: This is THE central dispatch mechanism in Flask. Static analysis cannot resolve which function handles which route.  
**Mitigation**: Type-flow analysis or integration with runtime profiling.

### Metaclasses
```python
class MethodViewType(type):
    def __init__(cls, name, bases, d):
        # Methods are injected here at class creation time
```
**Impact**: Method resolution order is modified at metaclass `__init__` time. Static analysis sees the class but not the injected methods.

---

## 2. Unresolved External References

When CodeGraph indexes a project, it only sees the project's own source code. Calls to external libraries produce **unresolved edges**.

| Project | Unresolved Edge % | Primary Cause |
|---------|-------------------|---------------|
| Flask | ~8% | Werkzeug, Jinja2, Click |
| Django (est.) | ~12% | Database drivers, middleware |
| FastAPI (est.) | ~10% | Starlette, Pydantic |

**Mitigation**: Index the dependency source code alongside the project. This increases index size but resolves most cross-library edges.

---

## 3. Conditional & Lazy Imports

```python
try:
    import ujson as json  # Fast path
except ImportError:
    import json           # Fallback

def expensive():
    import heavy_module   # Lazy import inside function body
```

**Current behavior**: Only top-level imports are captured. Imports inside `try/except`, `if/else`, or function bodies are **missed**.  
**Impact**: Low (~2-3% of imports in typical projects).  
**Fix complexity**: Medium — requires extending the parser to walk conditional bodies.

---

## 4. Decorator Side Effects

```python
@app.route('/api/data')  # Side effect: registers URL rule
@login_required           # Side effect: wraps function
def get_data():
    pass
```

**Current behavior**: Decorators are extracted as metadata, but their **side effects** (route registration, function wrapping) are not modeled in the graph.  
**Impact**: CodeGraph knows `get_data` has a `@app.route` decorator, but cannot trace the route registration call chain through the decorator.

---

## 5. Scalability Characteristics

| Codebase Size | Nodes | Edges | Index Time | Query Latency |
|---------------|-------|-------|------------|---------------|
| Small (<5K LoC) | ~100 | ~200 | <0.5s | <10ms |
| Medium (5-50K LoC) | ~500-2K | ~1-5K | 1-3s | 10-50ms |
| Large (50-200K LoC) | ~2-10K | ~5-20K | 3-10s | 50-200ms |
| Very Large (>200K LoC) | ~10K+ | ~20K+ | 10-30s | 200ms-1s |

**Bottleneck**: Community detection (Leiden algorithm) becomes the dominant cost above ~5K nodes. Graph traversal itself remains fast (BFS/DFS is O(V+E)).

---

## 6. Language-Specific Gaps

### Python
- ✅ Classes, functions, methods, imports, inheritance, decorators, calls
- ⚠️ Comprehension-internal definitions (rare)
- ❌ Dynamic attribute access, exec/eval, runtime imports

### TypeScript  
- ✅ Classes, functions, imports, type references
- ⚠️ Complex generic type resolution
- ❌ Runtime module loading (`require()` with variables)

---

## 7. Compared to Runtime Analysis

| Aspect | CodeGraph (Static) | Runtime Profiler |
|--------|-------------------|------------------|
| **Coverage** | All code paths | Only executed paths |
| **Accuracy** | Approximate (no dynamic) | Exact (what ran) |
| **Cost** | Zero (no execution) | High (instrumentation) |
| **Safety** | Safe (read-only) | Risky (must execute code) |
| **Dynamic dispatch** | ❌ Cannot resolve | ✅ Fully resolved |

**Ideal approach**: Static graph (CodeGraph) + runtime instrumentation = complete picture.

---

## Summary

CodeGraph is strongest on:
- **Deterministic structural queries** (call chains, inheritance, imports)
- **Exhaustive impact analysis** (every edge, every path)
- **Zero-cost indexing** (no LLM, no execution)

CodeGraph is weakest on:
- **Dynamic dispatch** (~5-10% of calls in Python)
- **Runtime-only patterns** (monkey-patching, metaclass injection)
- **Cross-library boundaries** (unless dependencies are indexed)
