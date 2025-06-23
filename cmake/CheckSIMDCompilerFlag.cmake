
file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-mavx512f"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp "#include <immintrin.h>\nint main() { __m512i a, b; b = _mm512_popcnt_epi64(a); return 0; }")
try_compile(COMPILER_AVX512VPOPCNTDQ_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp
    COMPILE_DEFINITIONS "-mavx512vpopcntdq"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(COMPILER_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-mavx2 -mfma"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-mavx"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(COMPILER_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-msse"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp "#include <arm_neon.h>\nint main() { float32x4_t a, b; a = vdupq_n_f32(1.0f); b = vdupq_n_f32(2.0f); a = vaddq_f32(a, b); return 0; }")
try_compile(COMPILER_NEON_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_neon
    ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp
    COMPILE_DEFINITIONS "-march=armv8-a"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp "#include <immintrin.h>\nint main() { __m512i a, b; b = _mm512_popcnt_epi64(a); return 0; }")
try_compile(COMPILER_AVX512VPOPCNTDQ_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp
    COMPILE_DEFINITIONS "-mavx512vpopcntdq"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(RUNTIME_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp "#include <arm_neon.h>\nint main() { float32x4_t a, b; a = vdupq_n_f32(1.0f); b = vdupq_n_f32(2.0f); a = vaddq_f32(a, b); return 0; }")
try_compile(RUNTIME_NEON_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_neon
    ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp
    COMPILE_DEFINITIONS "-march=armv8-a"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

# determine which instructions can be package into distribution
set (COMPILER_SUPPORTED "compiler support instructions: ")
if (COMPILER_SSE_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} SSE")
endif ()
if (COMPILER_AVX_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX")
endif ()
if (COMPILER_AVX2_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX2")
endif ()
if (COMPILER_AVX512_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX512")
endif ()
if (COMPILER_AVX512VPOPCNTDQ_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX512VPOPCNTDQ")
endif ()
if (COMPILER_NEON_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} NEON")
endif ()
message (${COMPILER_SUPPORTED})

# RUNTIME just output for debugging
set (RUNTIME_SUPPORTED "runtime support instructions: ")
if (RUNTIME_SSE_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} SSE")
endif ()
if (RUNTIME_AVX_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX")
endif ()
if (RUNTIME_AVX2_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX2")
endif ()
if (RUNTIME_AVX512_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX512")
endif ()
if (COMPILER_AVX512VPOPCNTDQ_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX512VPOPCNTDQ")
endif ()
if (RUNTIME_NEON_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} NEON")
endif ()
message (${RUNTIME_SUPPORTED})

# important distribution logic:
#       (not force disable) and (compiler support) and (lower instructions contained)
set (DIST_CONTAINS_INSTRUCTIONS "distribution contain instructions: ")
if (NOT DISABLE_SSE_FORCE AND COMPILER_SSE_SUPPORTED)
  set (DIST_CONTAINS_SSE ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} SSE")
endif ()
if (NOT DISABLE_AVX_FORCE AND COMPILER_AVX_SUPPORTED AND DIST_CONTAINS_SSE)
  set (DIST_CONTAINS_AVX ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX")
endif ()
if (NOT DISABLE_AVX2_FORCE AND COMPILER_AVX2_SUPPORTED AND DIST_CONTAINS_AVX)
  set (DIST_CONTAINS_AVX2 ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX2")
endif ()
if (NOT DISABLE_AVX512_FORCE AND COMPILER_AVX512_SUPPORTED AND DIST_CONTAINS_AVX2)
  set (DIST_CONTAINS_AVX512 ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX512")
endif ()
if (NOT DISABLE_AVX512VPOPCNTDQ_FORCE AND COMPILER_AVX512VPOPCNTDQ_SUPPORTED AND DIST_CONTAINS_AVX512)
  set (DIST_CONTAINS_AVX512VPOPCNTDQ ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX512VPOPCNTDQ")
endif ()
if (NOT DISABLE_NEON_FORCE AND COMPILER_NEON_SUPPORTED)
  set (DIST_CONTAINS_NEON ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} NEON")
endif ()
message (${DIST_CONTAINS_INSTRUCTIONS})
