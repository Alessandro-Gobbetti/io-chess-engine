cmake_minimum_required(VERSION 3.20)
project(ChessFeatureExtractor LANGUAGES CXX)

# --- 1. C++ Standard (std=c++20) ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 2. Compiler Flags (-O3 -march=native) ---
# Note: In CMake, -O3 is usually handled by CMAKE_BUILD_TYPE=Release.
# However, to force it exactly like your Makefile regardless of build type:
add_compile_options(-O3 -march=native -Wall)

# --- 3. Dependencies (-pthread) ---
find_package(Threads REQUIRED)

# --- 4. Include Directories (-I...) ---
# These are added globally for all targets in this file
include_directories(
    include
    lib/chess
    lib/progressbar
)

# --- 5. Source Files (wildcard src/*.cpp) ---
# NOTE: While explicit lists are safer, this mimics your Makefile's wildcard behavior.
file(GLOB SOURCES "src/*.cpp")

# --- 6. The Executable Target ---
add_executable(board_to_features ${SOURCES})

# Link the threading library found in step 3
target_link_libraries(board_to_features PRIVATE Threads::Threads)

# --- 7. Output Directory (bin/) ---
# This ensures the binary ends up in ./bin/ just like your Makefile
set_target_properties(board_to_features PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)