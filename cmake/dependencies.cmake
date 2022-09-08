set(CMAKE_FOLDER ThirdParty)

include(FetchContent)

# Google Abseil C++ Library
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG 20220623.0
)

# tensorflow-directml C API
if(WIN32)
    FetchContent_Declare(
        tensorflow
        URL https://github.com/microsoft/tensorflow-directml/releases/download/v1.15.7/libtensorflow-win-x64.zip
        URL_HASH SHA256=9ccc63ba035749c378cd1b99f5106f0f03534cd01bbd6674b36426544af0924d
    )
else()
    FetchContent_Declare(
        tensorflow
        URL https://github.com/microsoft/tensorflow-directml/releases/download/v1.15.7/libtensorflow-linux-x64.zip
        URL_HASH SHA256=fe5d6d08e4cb144053c05394876f91b561ef9489bcd6a7d376ce3c091dbd006c
    )
endif()

FetchContent_Declare(
    squeezenet_model
    URL https://github.com/oracle/graphpipe/raw/v1.0.0/docs/models/squeezenet.pb
    URL_HASH SHA256=5922a640a9e23978e9aef1bef16aaa89cc801bc3a30f4766a8c8fd4e1c6d81bc
    DOWNLOAD_NO_EXTRACT TRUE
)

# Download and extract dependencies.
FetchContent_MakeAvailable(
    abseil
    tensorflow
    squeezenet_model
)

add_library(tensorflow_libs INTERFACE)
target_link_libraries(
    tensorflow_libs
    INTERFACE
    $<$<BOOL:${WIN32}>:${tensorflow_SOURCE_DIR}/lib/tensorflow.lib>
    $<$<BOOL:${UNIX}>:${tensorflow_SOURCE_DIR}/lib/libtensorflow.so>
    $<$<BOOL:${UNIX}>:${tensorflow_SOURCE_DIR}/lib/libtensorflow.so>
)

set(CMAKE_FOLDER "")