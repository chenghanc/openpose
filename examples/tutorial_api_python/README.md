# Python API Examples
See the [OpenPose Python API doc](../../doc/03_python_api.md) for more details on this folder.

This folder provides examples to the basic OpenPose Python API. The analogous C++ API is exposed in [examples/tutorial_api_cpp/](../tutorial_api_cpp/).

---

# How to run Python API with conda environment

- Enable conda environment: `source activate v5`
- To compile, enable `BUILD_PYTHON` in CMake-gui, or run `cmake -DBUILD_PYTHON=ON ..` from your build directory
- Run `cmake -DPYTHON_EXECUTABLE=~/anaconda3/envs/v5/bin/python3.8 -DPYTHON_LIBRARY=~/anaconda3/envs/v5/lib/libpython3.8.so ..` and `make` from your build directory
  - A `pyopenpose.cpython-38-x86_64-linux-gnu.so` file will be created in `build/python/openpose` directory
  - Note that `*-38-*.so` filename has to match your python version
