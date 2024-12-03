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
$ ./yolov7_tiny -p ../data/horses.jpg  -m ../data/yolov7_tiny_int8.adla
```
