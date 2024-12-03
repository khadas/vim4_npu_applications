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
$ ./yolov3 -p ../data/1080p.bmp  -m ../data/det_yolov3_int8.adla
```
