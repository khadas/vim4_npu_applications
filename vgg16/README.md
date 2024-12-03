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
$ ./vgg16 -p ../data/airplane.jpeg  -m ../data/vgg16_int8.adla
```
