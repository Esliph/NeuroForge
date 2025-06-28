#pragma once

#ifndef LOG_LEVEL
#define LOG(x)
#else
#include <iostream>
#define LOG(x) std::cout << x << std::endl;
#endif
