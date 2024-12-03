# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

This demo integrates retinaface and facenet.

```shell
# Generate feature library
$ ./face_recognition -M ../data/model/retinaface_int8.adla -m ../data/model/facenet_int8.adla -p 1

# Identification
$ ./face_recognition -M ../data/model/retinaface_int8.adla -m ../data/model/facenet_int8.adla -p ../data/img/lin_1.jpg
```

When you generate feature library, please make sure only one face in picture.