# TensorFlow-DirectML-C-API-Samples <!-- omit in toc -->

TensorFlow-DirectML-C-API-Samples contains samples that show how to use the TensorFlow C API with the [tensorflow-directml](https://github.com/microsoft/tensorflow-directml) library.

## Getting Started

### Build and run the samples on Windows
1. Install the latest version of [CMake](https://cmake.org/download/) (CMake >= 3.20 is required)
2. Install the latest version of the [Visual Studio build tools](https://visualstudio.microsoft.com/downloads/?q=build+tools) or any other versions of Visual Studio
3. Run the following commands:

```
.\build.ps1
cd build
.\infer_squeezenet_model.exe
```

### Build and run the samples on WSL (Windows Subsystem for Linux)
1. Install the latest version of [CMake](https://cmake.org/download/) (CMake >= 3.20 is required)
2. Install the latest version of clang (e.g. on Ubuntu `apt update && apt install clang`)
3. Run the following commands:

```sh
./build.sh
cd build
./infer_squeezenet_model
```