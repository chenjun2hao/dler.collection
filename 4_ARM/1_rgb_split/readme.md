## 采用ARM NEON指令做图片rgb通道分离


采用neon intrinsics做加速

## 1. 编译
```bash
mkdir build && cd build
cmake ..
make -j4
```

## 2. 测试
```
./rgb

输入:
neon cost time is:6ms
opencv cost time is:5ms
own cost time is:24ms
```

这里可以将注释的代码uncomment，但是多运行几次`rgb` own cost time就会减少，暂时还不知道为什么