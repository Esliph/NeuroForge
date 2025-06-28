#pragma once

#include <iostream>

#define LOG_RESET_COLOR "\x1B[0m"
#define LOG_INFO_COLOR "\x1B[1;34m"
#define LOG_DEBUG_COLOR "\x1B[1;30m"
#define LOG_WARNING_COLOR "\x1B[1;33m"
#define LOG_ERROR_COLOR "\x1B[31m"

template <typename... Args>
void log(const char* level, const char* color, const char* fmt, Args... args) {
  std::printf("%s%s - ", color, level);
  std::printf(fmt, args...);
  std::printf("%s\n", LOG_RESET_COLOR);
}

#define LOG_MACRO_IMPL(level, color, fmt, ...) log(level, color, fmt, ##__VA_ARGS__)

#ifdef LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...) LOG_MACRO_IMPL("[INFO]", LOG_INFO_COLOR, fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...)
#endif

#ifdef LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) LOG_MACRO_IMPL("[DEBUG]", LOG_DEBUG_COLOR, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#ifdef LOG_LEVEL_WARNING
#define LOG_WARNING(fmt, ...) LOG_MACRO_IMPL("[WARNING]", LOG_WARNING_COLOR, fmt, ##__VA_ARGS__)
#else
#define LOG_WARNING(fmt, ...)
#endif

#ifdef LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...) LOG_MACRO_IMPL("[ERROR]", LOG_ERROR_COLOR, fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...)
#endif
