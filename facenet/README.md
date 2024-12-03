# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

```shell
# Generate feature library
$ ./facenet -p 1  -m ../data/model/facenet_int8.adla

# Identification
$ ./facenet -p ../data/img/lin_1.jpg  -m ../data/model/facenet_int8.adla
```
