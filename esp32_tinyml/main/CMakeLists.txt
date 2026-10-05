# main/CMakeLists.txt
idf_component_register(SRCS "main.cc"
                       INCLUDE_DIRS "."
                       REQUIRES "esp-tflite-micro" "esp_timer")

# TFLM requires specific C++ compiler flags (C++17 standard)
target_compile_options(${COMPONENT_LIB} PRIVATE "-std=gnu++17" "-fno-rtti" "-fno-exceptions" "-Wno-shadow" "-Wno-error=shadow")