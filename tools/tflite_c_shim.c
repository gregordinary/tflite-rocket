// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
/*
 * tflite_c_shim.c — provide the two TFLite C-API symbols the delegate references
 * (TfLiteIntArrayCreate / TfLiteIntArrayFree) for host interpreters that do NOT
 * export them.
 *
 * The delegate .so is headers-only: it expects TfLiteIntArrayCreate/Free to bind
 * at dlopen from the host interpreter (exactly like Mesa's libteflon.so). The
 * classic `tflite_runtime` / full `tensorflow` wheels export these as global
 * dynamic symbols, so the delegate resolves them with no help. LiteRT
 * (`ai_edge_litert`, the only wheel on recent Python) hides them, so the dlopen
 * fails with `undefined symbol: TfLiteIntArrayCreate`.
 *
 * Build this as a tiny shared object and LD_PRELOAD it before the interpreter:
 *   cc -shared -fPIC -I<mesa>/include tools/tflite_c_shim.c -o libtflite_cshim.so
 *   LD_PRELOAD=$PWD/libtflite_cshim.so python3 tools/run_delegate.py ...
 * The preloaded symbols are global, so the delegate (dlopen'd later) binds to
 * them. The definitions mirror tensorflow/lite/core/c/common.c byte-for-byte; the
 * delegate both creates and frees its own TfLiteIntArray (the node list handed to
 * ReplaceNodeSubsetsWithDelegateKernels), so a malloc/free pair is self-consistent
 * and never crosses an allocator boundary with the interpreter.
 */
#include <stdlib.h>
#include "tensorflow/lite/core/c/common.h"

size_t TfLiteIntArrayGetSizeInBytes(int size) {
    static TfLiteIntArray dummy;
    return sizeof(dummy) + sizeof(dummy.data[0]) * (size_t)size;
}

TfLiteIntArray *TfLiteIntArrayCreate(int size) {
    size_t alloc_size = TfLiteIntArrayGetSizeInBytes(size);
    if (alloc_size == 0)
        return NULL;
    TfLiteIntArray *ret = (TfLiteIntArray *)malloc(alloc_size);
    if (!ret)
        return ret;
    ret->size = size;
    return ret;
}

void TfLiteIntArrayFree(TfLiteIntArray *a) { free(a); }
