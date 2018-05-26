/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_TWEAKME_H
#define PT_TWEAKME_H

// Enable double precision (slower, disabled by default):
#define PT_DOUBLE_ENABLE 0

// Enable fused multiply add (faster, disabled by default):
#define PT_FMADD_ENABLE 0

// Enable internal loop unrolling (enabled by default for GCC):
#if defined(__GNUC__) || defined(__GNUG__)
    #define PT_LOOP_UNROLLING_ENABLE 1
#else
    #define PT_LOOP_UNROLLING_ENABLE 0
#endif

// Define libsimdpp arch:
#ifdef __arm__
    #define SIMDPP_ARCH_ARM_NEON_FLT_SP
#else
    #if PT_FMADD_ENABLE
        #define SIMDPP_ARCH_X86_AVX2
        #define SIMDPP_ARCH_X86_FMA3
    #else
        #define SIMDPP_ARCH_X86_AVX
    #endif
#endif

#endif
