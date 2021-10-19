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

# Test and Develop

- Duplicate and rename any of the existing sample files in `examples/tutorial_api_python/` within that folder and start building in there. Need to recompile OpenPose every time you make changes to your Python files so they are copied over the `build/` folder
- Change directory to `cd build/examples/tutorial_api_python`
- Run the code `python ../../../examples/tutorial_api_python/demo_video.py`

## References

---

<details><summary><b>CLICK ME</b> - Issues</summary>

- [Issues](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1623)

</details>

<details><summary><b>CLICK ME</b> - Compile OpenPose from Source</summary>

- [Prerequisites](https://github.com/chenghanc/openpose/blob/master/doc/installation/1_prerequisites.md#ubuntu-prerequisites)
- [Compiling](https://github.com/chenghanc/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source)

</details>

<details><summary><b>CLICK ME</b> - OpenPose Demo</summary>

```shell
./build/examples/openpose/openpose.bin --video examples/media/video.avi
./build/examples/openpose/openpose.bin --image_dir examples/media/

./build/examples/openpose/openpose.bin --video examples/media/video.avi --face --hand

./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output_jsons/
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_video output/result.avi

./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_images output_images/
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_images output_images/ --write_images_format jpg
./build/examples/openpose/openpose.bin --image_dir examples/media/ --write_images output_images/
./build/examples/openpose/openpose.bin --image_dir examples/media/ --write_images output_images/ --write_images_format jpg

./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_images output_images/ --display 0

./build/examples/openpose/openpose.bin --video examples/media/video.avi --disable_blending
./build/examples/openpose/openpose.bin --video examples/media/video.avi --num_gpu 1 --num_gpu_start 0
./build/examples/openpose/openpose.bin --video examples/media/video.avi --net_resolution "800x448" --scale_number 4 --scale_gap 0.25
```

```shell
# BODY_25B Model
./build/examples/openpose/openpose.bin --video examples/media/video.avi --model_pose BODY_25B
./build/examples/openpose/openpose.bin --video examples/media/video.avi --model_pose BODY_25B --net_resolution 448x448 --scale_number 4 --scale_gap 0.25
```

```shell
# Tracking
./build/examples/openpose/openpose.bin --video examples/media/video.avi --tracking 5 --number_people_max 1
```

</details>
