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
$ ./densenet_ctc -p ../data/KhadasTeam.png  -m ../data/densenet_ctc_int16.adla
```
