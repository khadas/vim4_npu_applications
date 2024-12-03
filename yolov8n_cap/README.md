# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

```shell
$ cd build
$ ./yolov8n_cap -m ../data/yolov8n_int8.adla -d X
```

**x**: the number for you camera device. such as `/dev/video0`, `x` is `0`.