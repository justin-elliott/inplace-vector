# jell::inplace_vector

An implementation of the C++26 [inplace_vector](https://en.cppreference.com/w/cpp/container/inplace_vector.html) container.

## Building

Building the project requires clang and libc++ version 20 or later.

```sh
mkdir build
cd build
cmake ..
make
```

## Testing

```sh
cd build
make && make test
```