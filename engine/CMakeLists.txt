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

# Source Files
set(FATHOM_SOURCES
    external/fathom/src/tbprobe.c
)
set_source_files_properties(${FATHOM_SOURCES} PROPERTIES LANGUAGE C)

set(ENGINE_SOURCES
    src/main.cpp
    src/eval/Evaluator.cpp
    src/eval/EvalContext.cpp
    src/eval/EvalContextMoE.cpp
    src/eval/SimpleEvalContext.cpp
    src/search/TT.cpp
    src/search/Negamax.cpp
    src/search/TimeManager.cpp
    src/search/MCTS.cpp
    src/uci/UCI.cpp
    src/book/polyglot_book.cpp
    src/book/polyglot_keys.cpp
    src/tablebases/Tablebase.cpp
    ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
    ${FATHOM_SOURCES}
)

# --- EMSCRIPTEN BUILD ---
if(EMSCRIPTEN)
    message(STATUS "Configuring for Emscripten (WASM)")

    # Optimization flags (WASM)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -fexceptions -msimd128 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -flto -fexceptions -msimd128 -DNDEBUG")

    # Model path for embedding
    set(MODEL_PATH "${CMAKE_SOURCE_DIR}/../training/onnx/light_moe_64f4b")

    # Linker Flags (WASM) - Configured for Web Worker loading
    set(EMSCRIPTEN_LINK_FLAGS
        "-O3 "                       # Optimize
        "-flto "                     # Link Time Optimization
        "-msimd128 "                 # Enable SIMD
        "-s WASM=1 "                 # Output WASM
        "-s ALLOW_MEMORY_GROWTH=1 "  # Allow memory to grow
        "-s INITIAL_MEMORY=536870912 " # Start with 512MB
        "-s MAXIMUM_MEMORY=4GB "     # Cap memory at 4GB
        "-s USE_PTHREADS=1 "         # Enable threads
        "-s PROXY_TO_PTHREAD=1 "     # Move main() to a thread to prevent blocking the message loop
        "-s PTHREAD_POOL_SIZE=8 "    # Pre-allocate threads
        "-s EXIT_RUNTIME=0 "         # Keep runtime alive
        "-s INVOKE_RUN=0 "           # Do not run main() automatically
        "-s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"callMain\"]' "
        "-s ENVIRONMENT='web,worker,node' " # Target browser/worker/node environment
        "-s MODULARIZE=1 "           # Export as ES module factory function
        "-s EXPORT_NAME='createChessEngine' " # Factory function name
        "-s STACK_SIZE=5242880 "     # 5MB Stack for main thread
        "-s DEFAULT_PTHREAD_STACK_SIZE=1048576 " # 1MB Stack for threads
        "--embed-file ${MODEL_PATH}@/model " # Embed ONNX model at /model
        "--embed-file ${CMAKE_SOURCE_DIR}/src/book/Perfect2023.bin@/book/Perfect2023.bin " # Embed Book
    )
    string(CONCAT CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} " ${EMSCRIPTEN_LINK_FLAGS})

    # ONNX Runtime Paths (WASM)
    set(ONNX_ROOT "${CMAKE_SOURCE_DIR}/../eval_model_onnx/onnxruntime")
    set(ONNX_BUILD "${ONNX_ROOT}/build/Linux/Release")
    set(ONNX_INCLUDE "${ONNX_ROOT}/include")
    set(FLATBUFFERS_INCLUDE "${ONNX_BUILD}/_deps/flatbuffers-src/include")

    include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/../preprocessing/include
        ${CMAKE_SOURCE_DIR}/../preprocessing/lib/chess
        ${CMAKE_SOURCE_DIR}/external/fathom/src
        ${ONNX_INCLUDE}
        ${FLATBUFFERS_INCLUDE} 
        "${ONNX_INCLUDE}/onnxruntime/core/session"
        "${ONNX_INCLUDE}/onnxruntime/core/providers/cpu"
        "${ONNX_INCLUDE}/onnxruntime/core/framework"
    )

    # Static Libraries (WASM)
    set(ONNX_LIBS
        "${ONNX_BUILD}/libonnxruntime_webassembly.a"
        "${ONNX_BUILD}/libonnxruntime_session.a"
        "${ONNX_BUILD}/libonnxruntime_providers.a"
        "${ONNX_BUILD}/libonnxruntime_framework.a"
        "${ONNX_BUILD}/libonnxruntime_optimizer.a"
        "${ONNX_BUILD}/libonnxruntime_graph.a"
        "${ONNX_BUILD}/libonnxruntime_mlas.a"
        "${ONNX_BUILD}/libonnxruntime_common.a"
        "${ONNX_BUILD}/libonnxruntime_flatbuffers.a"
        "${ONNX_BUILD}/libonnxruntime_util.a" 
    )

    add_executable(chess_engine ${ENGINE_SOURCES})
    target_link_libraries(chess_engine PRIVATE ${ONNX_LIBS})
    set_target_properties(chess_engine PROPERTIES SUFFIX ".js")

    # Feature-only WASM (no ONNX, no pthreads)
    add_executable(feature_wasm
        src/feature_api.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
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

    # IPO/LTO
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()

    # Optimization flags (Native)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native -fno-plt -funroll-loops -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

    # ONNX Runtime (Native dynamic linking)
    set(ONNX_VERSION "1.23.0")
    set(USE_GPU "auto" CACHE STRING "Use GPU ONNX Runtime: auto/on/off")

    if(APPLE)
        set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../eval_model_onnx/onnxruntime-osx-arm64-${ONNX_VERSION}")
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    else()
        if(USE_GPU STREQUAL "auto")
            if(EXISTS "${CMAKE_SOURCE_DIR}/../eval_model_onnx/onnxruntime-linux-x64-gpu-${ONNX_VERSION}")
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
            set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../eval_model_onnx/onnxruntime-linux-x64-gpu-${ONNX_VERSION}")
        else()
            set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../eval_model_onnx/onnxruntime-linux-x64-${ONNX_VERSION}")
        endif()
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()

    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    
    # RPATH
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
    set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")

    # Includes
    if(APPLE)
        include_directories("${MACOS_SDK_PATH}/usr/include/c++/v1")
    endif()
    include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/../preprocessing/include
        ${CMAKE_SOURCE_DIR}/../preprocessing/lib/chess
        ${CMAKE_SOURCE_DIR}/external/fathom/src
        ${ONNXRUNTIME_INCLUDE_DIR}
    )

    # Main Engine
    add_executable(chess_engine ${ENGINE_SOURCES})
    target_link_libraries(chess_engine ${ONNXRUNTIME_LIB} pthread)

    # Additional Tools
    add_executable(benchmark_eval 
        src/tools/benchmark_eval.cpp
        src/eval/Evaluator.cpp src/eval/EvalContext.cpp src/eval/EvalContextMoE.cpp src/eval/SimpleEvalContext.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
    )
    target_link_libraries(benchmark_eval ${ONNXRUNTIME_LIB} pthread)

    add_executable(eval_raw 
        src/tools/eval_raw.cpp
        src/eval/Evaluator.cpp src/eval/EvalContext.cpp src/eval/EvalContextMoE.cpp src/eval/SimpleEvalContext.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
    )
    target_link_libraries(eval_raw ${ONNXRUNTIME_LIB} pthread)

    add_executable(compare_evals 
        src/tools/compare_evals.cpp
        src/eval/Evaluator.cpp src/eval/EvalContext.cpp src/eval/EvalContextMoE.cpp src/eval/SimpleEvalContext.cpp
        ${CMAKE_SOURCE_DIR}/../preprocessing/src/FeatureExtractor.cpp
    )
    target_link_libraries(compare_evals ${ONNXRUNTIME_LIB} pthread)

endif()
