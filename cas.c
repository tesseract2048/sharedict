#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>

int cas_int(int *addr, int oldval, int newval) {
    return __sync_bool_compare_and_swap(addr, oldval, newval);
}
