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
$ ./yolov8n_pose -p ../data/bus.jpg  -m ../data/yolov8n_pose_int8.adla
```
