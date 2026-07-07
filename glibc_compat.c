// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
//
// glibc portability shim — opt-in via -DTFLITE_ROCKET_PORTABLE_GLIBC=ON.
//
// glibc 2.38 redirected strtol/fscanf to versioned __isoc23_* symbols, so a delegate
// compiled on a glibc >= 2.38 host (e.g. Debian trixie/forky) imports
// __isoc23_strtol / __isoc23_fscanf at GLIBC_2.38 and then fails to load on an older
// runtime — notably the Frigate container (Debian 12 bookworm, glibc 2.36). Those are
// the only > 2.36 symbols the delegate pulls in.
//
// This unit is compiled as C11 (-std=gnu11, so its OWN strtol/fscanf are NOT
// redirected and stay the classic GLIBC_2.17 symbols) and defines the two __isoc23_*
// entry points as thin forwarders. It is linked INTO libtflite_rocket.so with hidden
// visibility, so:
//   * the delegate's (and the static driver lib's) references to __isoc23_* resolve
//     to these definitions at link time — the .so no longer imports them from glibc,
//   * the symbols are not exported, so they cannot interpose on a newer host's libc,
//   * the .so's glibc floor drops to 2.34, which loads on bookworm.
// The forwarders are behaviourally identical for the integer option / sysfs parsing
// the delegate does; the C23 additions (e.g. 0b binary literals) are unused.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((visibility("hidden")))
long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

__attribute__((visibility("hidden")))
int __isoc23_fscanf(FILE *stream, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    int r = vfscanf(stream, format, ap);
    va_end(ap);
    return r;
}
