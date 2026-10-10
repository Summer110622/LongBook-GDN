cmake_minimum_required(VERSION 3.21)
project(Longbook27 VERSION 0.1.0 LANGUAGES C CXX)

option(LONGBOOK27_CUDA "Build the optional CUDA backend" OFF)
option(LONGBOOK27_METAL "Build the optional Metal backend" OFF)
option(LONGBOOK27_NATIVE "Optimize CPU code for the build machine" OFF)
option(LONGBOOK27_OPENMP "Use a locally installed OpenMP runtime" OFF)

if(LONGBOOK27_CUDA AND LONGBOOK27_METAL)
    message(FATAL_ERROR "Select at most one optional GPU backend")
endif()
if(LONGBOOK27_METAL AND NOT APPLE)
    message(FATAL_ERROR "The Metal backend requires an Apple platform and local Xcode tools")
endif()
if(WIN32 AND LONGBOOK27_OPENMP)
    message(FATAL_ERROR "This distribution disables OpenMP on Windows to avoid upstream runtime downloads")
endif()

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/vendor_engine.py" --verify
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE LONGBOOK27_VENDOR_RESULT
    OUTPUT_VARIABLE LONGBOOK27_VENDOR_OUTPUT
    ERROR_VARIABLE LONGBOOK27_VENDOR_ERROR
)
if(NOT LONGBOOK27_VENDOR_RESULT EQUAL 0)
    message(FATAL_ERROR "Pinned engine verification failed:\n${LONGBOOK27_VENDOR_ERROR}")
endif()
string(STRIP "${LONGBOOK27_VENDOR_OUTPUT}" LONGBOOK27_VENDOR_OUTPUT)
message(STATUS "${LONGBOOK27_VENDOR_OUTPUT}")

file(READ "${PROJECT_SOURCE_DIR}/engine.lock.json" LONGBOOK27_ENGINE_LOCK)
string(JSON LONGBOOK27_ENGINE_COMMIT GET "${LONGBOOK27_ENGINE_LOCK}" source commit)

# All executable code is built from the shipped source snapshot. No FetchContent,
# server, converters, or model download is part of configuration or compilation.
foreach(option_name IN ITEMS
    LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_TOOLS
    LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_USE_PREBUILT_UI
    LLAMA_BUILD_MTMD LLAMA_OPENSSL LLAMA_SUBPROCESS LLAMA_LLGUIDANCE
    LLAMA_TOOLS_INSTALL LLAMA_TESTS_INSTALL LLAMA_USE_SYSTEM_GGML
    BUILD_SHARED_LIBS GGML_BACKEND_DL GGML_CPU_ALL_VARIANTS GGML_CPU_KLEIDIAI
    GGML_CPU_HBM GGML_BLAS GGML_RPC GGML_CANN GGML_HIP GGML_MUSA GGML_SYCL
    GGML_VULKAN GGML_WEBGPU GGML_ZDNN GGML_OPENCL GGML_HEXAGON GGML_ZENDNN
    GGML_OPENVINO GGML_ET GGML_VIRTGPU
)
    set(${option_name} OFF CACHE BOOL "Disabled in the curated Longbook27 engine" FORCE)
endforeach()
set(LLAMA_BUILD_NUMBER 0 CACHE STRING "Pinned source snapshot" FORCE)
set(LLAMA_BUILD_COMMIT "${LONGBOOK27_ENGINE_COMMIT}" CACHE STRING "Pinned engine SHA" FORCE)
set(GGML_CPU ON CACHE BOOL "CPU backend is required" FORCE)
set(GGML_CUDA "${LONGBOOK27_CUDA}" CACHE BOOL "Longbook27 optional CUDA backend" FORCE)
set(GGML_METAL "${LONGBOOK27_METAL}" CACHE BOOL "Longbook27 optional Metal backend" FORCE)
set(GGML_NATIVE "${LONGBOOK27_NATIVE}" CACHE BOOL "Optimize for local CPU" FORCE)
set(GGML_OPENMP "${LONGBOOK27_OPENMP}" CACHE BOOL "Use local OpenMP" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal kernels in the client library" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

add_executable(longbook27 src/main.cpp src/client.cpp)
target_compile_features(longbook27 PRIVATE cxx_std_17)
set_target_properties(longbook27 PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(longbook27 PRIVATE llama)
target_compile_definitions(longbook27 PRIVATE
    LONGBOOK27_VERSION="${PROJECT_VERSION}"
    LONGBOOK27_ENGINE_COMMIT="${LONGBOOK27_ENGINE_COMMIT}"
)
if(MSVC)
    target_compile_options(longbook27 PRIVATE /W4 /utf-8)
else()
    target_compile_options(longbook27 PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME native-help COMMAND longbook27 --help)
    add_test(NAME native-version COMMAND longbook27 --version)
    add_test(NAME python-unit-and-native
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests -v
    )
    set_tests_properties(python-unit-and-native PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        ENVIRONMENT "LONGBOOK27_BIN=$<TARGET_FILE:longbook27>;PYTHONDONTWRITEBYTECODE=1"
        TIMEOUT 180
    )
    add_test(NAME engine-provenance
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/vendor_engine.py" --verify
    )
endif()

install(TARGETS longbook27 RUNTIME DESTINATION bin)
install(FILES LICENSE DESTINATION share/longbook27)
install(FILES third_party/llama.cpp/LICENSE third_party/llama.cpp/AUTHORS
    DESTINATION share/longbook27/engine)
install(DIRECTORY third_party/llama.cpp/licenses/
    DESTINATION share/longbook27/engine/licenses)
install(FILES engine.lock.json third_party/llama.cpp/LONGBOOK27_VENDOR_AUDIT.json
    DESTINATION share/longbook27/engine)

# Keep additional attributions from source headers with installed binaries.
# Exact source contents are already covered by the pinned snapshot manifest.
file(STRINGS "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp"
    LONGBOOK27_SGEMM_NOTICE LIMIT_COUNT 21)
string(JOIN "\n" LONGBOOK27_SGEMM_NOTICE ${LONGBOOK27_SGEMM_NOTICE})
file(STRINGS "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-cpu/ops.cpp"
    LONGBOOK27_YARN_NOTICE REGEX "^// (YaRN algorithm|MIT licensed[.] Copyright)")
string(JOIN "\n" LONGBOOK27_YARN_NOTICE ${LONGBOOK27_YARN_NOTICE})
file(READ "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/LICENSE" LONGBOOK27_ENGINE_LICENSE)
file(WRITE "${PROJECT_BINARY_DIR}/ENGINE-THIRD-PARTY-NOTICES.txt"
    "Additional notices retained from the pinned engine source\n\n"
    "Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp\n${LONGBOOK27_SGEMM_NOTICE}\n\n"
    "Source: ggml/src/ggml-cpu/ops.cpp; same YaRN notice is present in CUDA and Metal kernels\n"
    "${LONGBOOK27_YARN_NOTICE}\n\nMIT license text shipped with the engine:\n${LONGBOOK27_ENGINE_LICENSE}")
install(FILES "${PROJECT_BINARY_DIR}/ENGINE-THIRD-PARTY-NOTICES.txt"
    DESTINATION share/longbook27/engine)
