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
$ ./retinaface -p ../data/timg.jpg -m ../data/retinaface_int8.adla
```
