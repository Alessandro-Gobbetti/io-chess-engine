cmake_minimum_required(VERSION 3.14)

project(ChessEngine LANGUAGES C CXX)

# Configure sysroot BEFORE enabling languages to ensure compilers work? 
# Actually, project() enables languages. 
# For MacOS native we needed sysroot before.
# But for Emscripten we need toolchain first.
# Standard practice: define project, then check variables. Note: CMAKE_OSX_SYSROOT might need to be set before project on some cmake versions/generators if relying on it for compiler check?
# However, if we simply move project() up, Emscripten works.
# If MacOS breaks, we can revisit. Current priority is WASM.


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type - Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_ONNX_TOOLS "Build ONNX-based benchmark/parity tools" OFF)

# Source Files
set(FATHOM_SOURCES
    external/fathom/src/tbprobe.c
)
set_source_files_properties(${FATHOM_SOURCES} PROPERTIES LANGUAGE C)

set(ENGINE_SOURCES
    src/main.cpp
    src/eval/Evaluator.cpp
    src/eval/EvalContextMoECache.cpp
    src/eval/SimpleEvalContext.cpp
    src/search/TT.cpp
    src/search/Negamax.cpp
    src/search/MovePicker.cpp
    src/search/TablebaseFallback.cpp
    src/search/TimeManager.cpp
    src/search/MCTS.cpp
    src/uci/UCI.cpp
    src/uci/UciOptions.cpp
    src/platform/WasmSupport.cpp
    src/book/polyglot_book.cpp
    src/book/polyglot_keys.cpp
    src/tablebases/Tablebase.cpp
    ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
    ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
    ${FATHOM_SOURCES}
)

# --- EMSCRIPTEN BUILD ---
if(EMSCRIPTEN)
    message(STATUS "Configuring for Emscripten (WASM)")

    # Optimization flags (WASM)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -flto -fexceptions -msimd128 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -flto -fexceptions -msimd128 -DNDEBUG")

    # Build wasm engine without ONNX Runtime and require native bundle weights.
    set(IO_WASM_MODEL_PATH "${CMAKE_SOURCE_DIR}/../training/moe_cache/updated_loss_exported" CACHE PATH "Model directory embedded in wasm build")
    if(NOT EXISTS "${IO_WASM_MODEL_PATH}/native_weights.bin")
        message(FATAL_ERROR "IO_WASM_MODEL_PATH must contain native_weights.bin. Current: ${IO_WASM_MODEL_PATH}")
    endif()
    include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/../preprocessing/include
        ${CMAKE_SOURCE_DIR}/../preprocessing/lib/chess
        ${CMAKE_SOURCE_DIR}/external/fathom/src
    )

    set(WASM_EMBED_BOOK_ARGS)
    set(WASM_BOOK_FILE "${CMAKE_SOURCE_DIR}/src/book/Perfect2023.bin")
    if(EXISTS "${WASM_BOOK_FILE}")
        list(APPEND WASM_EMBED_BOOK_ARGS "SHELL:--embed-file ${WASM_BOOK_FILE}@/book/Perfect2023.bin")
    else()
        message(WARNING "WASM opening book not found at ${WASM_BOOK_FILE}; continuing without embedded book")
    endif()

    add_executable(io-chess ${ENGINE_SOURCES})
    target_link_options(io-chess PRIVATE
        "-O3"
        "-flto"
        "-msimd128"
        "-sWASM=1"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sINITIAL_MEMORY=536870912"
        "-sMAXIMUM_MEMORY=4GB"
        "-sUSE_PTHREADS=1"
        "-sPROXY_TO_PTHREAD=1"
        "-sPTHREAD_POOL_SIZE=8"
        "-sEXIT_RUNTIME=0"
        "-sINVOKE_RUN=0"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','callMain']"
        "-sENVIRONMENT=web,worker,node"
        "-sMODULARIZE=1"
        "-sEXPORT_NAME=createChessEngine"
        "-sSTACK_SIZE=5242880"
        "-sDEFAULT_PTHREAD_STACK_SIZE=1048576"
        "SHELL:--embed-file ${IO_WASM_MODEL_PATH}@/model"
        ${WASM_EMBED_BOOK_ARGS}
    )
    set_target_properties(io-chess PROPERTIES SUFFIX ".js")

    # Feature-only WASM (no ONNX, no pthreads)
    add_executable(feature_wasm
        src/feature_api.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
    )
    set_target_properties(feature_wasm PROPERTIES SUFFIX ".js")
    target_compile_definitions(feature_wasm PRIVATE FEATURE_WASM_MAIN=1)
    target_link_options(feature_wasm PRIVATE
        "-sUSE_PTHREADS=0"
        "-sPROXY_TO_PTHREAD=0"
        "-sPTHREAD_POOL_SIZE=0"
        "-sEXIT_RUNTIME=0"
        "-sINVOKE_RUN=0"
        "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAPF32']"
        "-sENVIRONMENT=web,worker"
        "-sMODULARIZE=1"
        "-sEXPORT_NAME=createFeatureWasm"
    )

# --- NATIVE BUILD ---
else()
    message(STATUS "Configuring for Native (Linux/macOS)")

    # Keep native incremental evaluator hot path optimized even if the
    # user configured a Debug build directory for diagnostics elsewhere.
    function(enable_native_perf_flags target_name)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${target_name} PRIVATE -O3 -march=native -ffast-math )
            target_link_options(${target_name} PRIVATE )
        endif()
    endfunction()

    # IPO/LTO
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()

    # Optimization flags (Native)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native -fno-plt -funroll-loops -DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native -funroll-loops -DNDEBUG")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

    # OpenMP (Optional, for non-Apple platforms)
    if(NOT APPLE)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            link_libraries(OpenMP::OpenMP_CXX)
            message(STATUS "OpenMP found and enabled")
        endif()
    endif()

    # Includes
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
        include_directories(SYSTEM "${MACOS_SDK_PATH}/usr/include/c++/v1")
    elseif(APPLE)
        message(STATUS "Non-Clang compiler on macOS detected; skipping libc++ include injection to avoid stdlib header conflicts")
    endif()
    include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/../preprocessing/include
        ${CMAKE_SOURCE_DIR}/../preprocessing/lib/chess
        ${CMAKE_SOURCE_DIR}/external/fathom/src
    )

    set(ONNX_CONTEXT_SOURCES
        src/tools/onnx_eval/EvalContext.cpp
        src/tools/onnx_eval/EvalContextMoE.cpp
    )
    set(ONNX_TOOL_EVALUATOR_SOURCES
        src/tools/OnnxEvaluator.cpp
        ${ONNX_CONTEXT_SOURCES}
    )

    if(BUILD_ONNX_TOOLS)
        message(STATUS "BUILD_ONNX_TOOLS=ON: enabling ONNX benchmark/parity tools")

        # ONNX Runtime (Native dynamic linking)
        set(ONNX_VERSION "1.23.0")
        set(USE_GPU "auto" CACHE STRING "Use GPU ONNX Runtime: auto/on/off")

        if(APPLE)
            set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-${ONNX_VERSION}")
            set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
        else()
            if(USE_GPU STREQUAL "auto")
                if(EXISTS "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}")
                    set(ONNX_RUNTIME_VARIANT "gpu")
                else()
                    set(ONNX_RUNTIME_VARIANT "")
                endif()
            elseif(USE_GPU STREQUAL "on" OR USE_GPU STREQUAL "ON" OR USE_GPU STREQUAL "true")
                set(ONNX_RUNTIME_VARIANT "gpu")
            else()
                set(ONNX_RUNTIME_VARIANT "")
            endif()

            if(ONNX_RUNTIME_VARIANT STREQUAL "gpu")
                set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}")
            else()
                set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-${ONNX_VERSION}")
            endif()
            set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
        endif()

        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

        # RPATH
        set(CMAKE_SKIP_BUILD_RPATH FALSE)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
        set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")

        function(enable_onnx_includes target_name)
            target_include_directories(${target_name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        endfunction()
    else()
        message(STATUS "BUILD_ONNX_TOOLS=OFF: skipping ONNX benchmark/parity tools")
    endif()

    # Main Engine (runtime only: native weights path, no ONNX runtime dependency)
    add_executable(io-chess ${ENGINE_SOURCES})
    target_link_libraries(io-chess pthread)
    enable_native_perf_flags(io-chess)

    # Additional Tools
    if(BUILD_ONNX_TOOLS)
        add_executable(benchmark_eval
            src/tools/benchmark_eval.cpp
            ${ONNX_TOOL_EVALUATOR_SOURCES}
            src/eval/Evaluator.cpp src/eval/EvalContextMoECache.cpp src/eval/SimpleEvalContext.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        )
        target_link_libraries(benchmark_eval ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(benchmark_eval)
        enable_native_perf_flags(benchmark_eval)

        add_executable(eval_raw
            src/tools/eval_raw.cpp
            ${ONNX_TOOL_EVALUATOR_SOURCES}
            src/eval/Evaluator.cpp src/eval/EvalContextMoECache.cpp src/eval/SimpleEvalContext.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        )
        target_link_libraries(eval_raw ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(eval_raw)
        enable_native_perf_flags(eval_raw)

        add_executable(compare_evals
            src/tools/compare_evals.cpp
            ${ONNX_TOOL_EVALUATOR_SOURCES}
            src/eval/Evaluator.cpp src/eval/EvalContextMoECache.cpp src/eval/SimpleEvalContext.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        )
        target_link_libraries(compare_evals ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(compare_evals)
        enable_native_perf_flags(compare_evals)

        add_executable(benchmark_factorized
            src/tools/benchmark_factorized.cpp
            ${ONNX_CONTEXT_SOURCES}
            src/eval/Evaluator.cpp src/eval/EvalContextMoECache.cpp src/eval/SimpleEvalContext.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        )
        target_link_libraries(benchmark_factorized ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(benchmark_factorized)
        enable_native_perf_flags(benchmark_factorized)
    endif()

    add_executable(benchmark_factorized_moe_cache
        src/tools/benchmark_factorized_moe_cache.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
    )
    target_link_libraries(benchmark_factorized_moe_cache pthread)
    target_compile_options(benchmark_factorized_moe_cache PRIVATE -O3 -march=native  -ffast-math)
    target_link_options(benchmark_factorized_moe_cache PRIVATE )

    add_executable(benchmark_factorized_moe_cache_quant
        src/tools/benchmark_factorized_moe_cache_quant.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
    )
    target_link_libraries(benchmark_factorized_moe_cache_quant pthread)
    target_compile_options(benchmark_factorized_moe_cache_quant PRIVATE -O3 -march=native  -ffast-math)
    target_link_options(benchmark_factorized_moe_cache_quant PRIVATE )

    if(BUILD_ONNX_TOOLS)
        add_executable(verify_model_new_test_parity
            src/tools/verify_model_new_test_parity.cpp
        )
        target_link_libraries(verify_model_new_test_parity ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(verify_model_new_test_parity)
        enable_native_perf_flags(verify_model_new_test_parity)

        add_executable(benchmark_factorized_moe_cache_parity
            src/tools/benchmark_factorized_moe_cache_parity.cpp
            src/tools/moe_cache_onnx_runner.cpp
            ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        )
        target_link_libraries(benchmark_factorized_moe_cache_parity ${ONNXRUNTIME_LIB} pthread)
        enable_onnx_includes(benchmark_factorized_moe_cache_parity)
        target_compile_options(benchmark_factorized_moe_cache_parity PRIVATE -O3 -march=native  -ffast-math)
        target_link_options(benchmark_factorized_moe_cache_parity PRIVATE )
    endif()

    add_executable(export_rich_features
        src/tools/export_rich_features.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
    )
    target_link_libraries(export_rich_features pthread)
    enable_native_perf_flags(export_rich_features)

endif()

if(NOT EMSCRIPTEN)
    add_executable(trace_tt 
        src/tools/trace_tt.cpp
        src/eval/Evaluator.cpp src/eval/EvalContextMoECache.cpp src/eval/SimpleEvalContext.cpp
        src/search/TT.cpp src/search/Negamax.cpp src/search/MovePicker.cpp src/search/TablebaseFallback.cpp src/search/TimeManager.cpp src/search/MCTS.cpp
        src/uci/UCI.cpp src/uci/UciOptions.cpp src/platform/WasmSupport.cpp src/book/polyglot_book.cpp src/book/polyglot_keys.cpp src/tablebases/Tablebase.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FactorizedFeatureExtractor.cpp
        ${FATHOM_SOURCES}
    )
    target_link_libraries(trace_tt pthread)
endif()
