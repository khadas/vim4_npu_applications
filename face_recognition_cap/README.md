# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

This demo integrates retinaface and facenet.

Before running, please run face_recognition to generate face_feature_lib and copy the library here.

```shell
# Identification
$ ./face_recognition -M ../data/model/retinaface_int8.adla -m ../data/model/facenet_int8.adla -d x
```

**x**: the number for you camera device. such as `/dev/video0`, `x` is `0`.