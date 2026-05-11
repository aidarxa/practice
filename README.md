# crystal-sycl

## JIT include path configuration

`AdaptiveCppCompiler` now requires explicit include directories for JIT compilation.
You can provide them in two ways:

1. **CMake defaults (recommended):**
   - `CRYSTAL_INCLUDE_DIR`
   - `CRYSTAL_DEPS_INCLUDE_DIR`
2. **Runtime environment variables (override/fallback):**
   - `CRYSTAL_INCLUDE_DIR`
   - `CRYSTAL_DEPS_INCLUDE_DIR`

The compiler validates both paths before calling `acpp` and fails early if a path is empty, missing, or not a directory.

### Configure with CMake

```bash
cmake -S . -B build \
  -DCRYSTAL_INCLUDE_DIR="$PWD/include" \
  -DCRYSTAL_DEPS_INCLUDE_DIR="$PWD/deps/include"
```

### Configure with environment variables

```bash
export CRYSTAL_INCLUDE_DIR="$PWD/include"
export CRYSTAL_DEPS_INCLUDE_DIR="$PWD/deps/include"
```

If constructor parameters and environment variables are both present, constructor parameters are used first.
