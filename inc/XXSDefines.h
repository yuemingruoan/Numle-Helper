#ifndef XXSLIBRARY_XXSDEFINES_H
#define XXSLIBRARY_XXSDEFINES_H

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    // Windows
    #define XXS_PLATFORM_WINDOWS
    #if defined(_WIN64)
        // Windows 64-bit
        #define XXS_PLATFORM_WINDOWS_64
    #else
        // Windows 32-bit
        #define XXS_PLATFORM_WINDOWS_32
    #endif
#elif defined(__APPLE__) && defined(__MACH__)
    // Apple platforms (macOS, iOS, etc.)
    #include <TargetConditionals.h>
    #if TARGET_OS_MAC == 1
        // macOS
        #define XXS_PLATFORM_MACOS
    #else
        // Other Apple platforms (iOS, etc.)
        #define XXS_PLATFORM_IOS
    #endif
#elif defined(__linux__)
    // Linux
    #define XXS_PLATFORM_LINUX
#elif defined(__unix__)
    // Other Unix-like systems
    #define XXS_PLATFORM_UNIX
#else
    // Unknown platform
    #error "Unknown platform detected!"
#endif

#endif //XXSLIBRARY_XXSDEFINES_H
