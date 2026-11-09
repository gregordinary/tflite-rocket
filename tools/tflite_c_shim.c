// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The tflite-rocket authors
/*
 * tflite_c_shim.c -- TfLiteIntArrayCreate and TfLiteIntArrayFree for host interpreters
 * that do not make them visible to a dlopen'd delegate.
 *
 * The delegate is built against the TFLite C headers alone and expects these two symbols
 * to bind at dlopen from the host interpreter, as Mesa's libteflon.so does. LiteRT
 * (`ai_edge_litert`) does not export them, and `tflite_runtime` loads its C extension
 * RTLD_LOCAL, so neither wheel makes them visible, and the dlopen fails with
 * `undefined symbol: TfLiteIntArrayCreate`. Preload this object and its symbols are in
 * the global scope before the delegate loads:
 *   cc -std=c11 -O2 -shared -fPIC tools/tflite_c_shim.c -o libtflite_cshim.so
 *   LD_PRELOAD=$PWD/libtflite_cshim.so python3 tools/run_delegate.py ...
 *
 * A TfLiteIntArray is an int count followed directly by that many ints. The delegate
 * creates every array it passes to these functions and frees it itself (the node list it
 * hands ReplaceNodeSubsetsWithDelegateKernels, which the interpreter copies), so the pair
 * only has to agree with itself, and both use the C library allocator.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* The layout of a TfLiteIntArray. */
struct shim_int_list {
    int count;
    int items[];
};

_Static_assert(offsetof(struct shim_int_list, items) == sizeof(int),
               "the ints must follow the count with no padding");
_Static_assert(sizeof(struct shim_int_list) == sizeof(int),
               "the header must be exactly one int");

#define SHIM_EXPORT __attribute__((visibility("default")))

/* NULL for a negative count, a count whose byte size does not fit size_t, or an
 * allocation failure. The entries start zeroed. */
SHIM_EXPORT struct shim_int_list *TfLiteIntArrayCreate(int size)
{
    if (size < 0)
        return NULL;
    const size_t count = (size_t)size;
    if (count > (SIZE_MAX - sizeof(struct shim_int_list)) / sizeof(int))
        return NULL;
    struct shim_int_list *list = calloc(1, sizeof(struct shim_int_list) + count * sizeof(int));
    if (list != NULL)
        list->count = size;
    return list;
}

SHIM_EXPORT void TfLiteIntArrayFree(struct shim_int_list *list)
{
    free(list);
}
