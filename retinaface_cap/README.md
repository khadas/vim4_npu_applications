# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

## MIPI Camera

```shell
$ cd build
$ ./retinaface_cap -m ../data/retinaface_int8.adla -t mipi -w 1920 -h 1080
```

## USB Camera
```shell
$ cd build
$ ./retinaface_cap -m ../data/retinaface_int8.adla -t usb -w 1920 -h 1080 -d 0
```

**TIP**: Replace 0 as the number for your camera device. Such as `/dev/video5`, it should be `-d 5`.