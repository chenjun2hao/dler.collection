#ifndef NET_H
#define NET_H

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#ifdef tx2
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxParserRuntime.h>
#define num 0
#else
#include <tensorrt/NvInfer.h>
#include <tensorrt/NvOnnxParser.h>
#include <tensorrt/NvOnnxParserRuntime.h>
#define num 1
#endif

#endif