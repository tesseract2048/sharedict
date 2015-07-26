'''
A cython version of dict, residented in shared memory which can be accessed by name.
Using marshal for serialization.
'''

from posix.fcntl cimport open
from posix.unistd cimport off_t
from cpython cimport bool
import os
import time
import psutil
import marshal
import mmh3

cdef extern from "unistd.h":
    int ftruncate(int fd, off_t length)
    int close(int fd)

cdef extern from "sys/types.h":
    enum:
        S_IRUSR
        S_IWUSR
        S_IRGRP
        S_IROTH

cdef extern from "sys/mman.h":
    void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset)
    int shm_open(char *name, int oflag, int mode)
    int shm_unlink(char *name)
    int munmap(void *addr, size_t length)
    enum:
        PROT_READ
        PROT_WRITE
        MAP_FILE
        MAP_SHARED
        O_RDONLY
        O_RDWR
        O_CREAT
        O_TRUNC
        O_EXCL

cdef extern from "sys/stat.h":
    cdef struct stat:
        off_t st_size
    int fstat(int fildes, stat *buf)

cdef extern from "errno.h":
    int errno
    enum:
        EEXIST

cdef extern from "cas.h":
    int cas_int(int *addr, int oldval, int newval)

cdef extern from "string.h":
    void *memcpy(void *dest, void *src, size_t n)
    void *memset(void *s, int c, size_t n)
    int memcmp(void *s1, void *s2, size_t n)

ctypedef struct HashEntry:
    int key_pos
    int value_pos

ctypedef struct Layout:
    size_t size
    int state
    int leader_pid
    int num_elements
    int num_entries
    size_t blob_start
    size_t blob_end
    HashEntry first_entry


STATE_PRE_INIT = 0  # layout is not yet initialized
STATE_INIT = 1      # layout is during initialization
STATE_FILL = 2      # layout is loading data
STATE_READY = 3     # layout is ready for reading

header_size = sizeof(Layout) - sizeof(HashEntry)
default_size = 65536
mode0644 = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH

'''
Double layout size.
'''
cdef void extend_layout(bytes name):
    cdef stat shmstat
    cdef bytes path = bytes("sharedict_%s_%s" % (job_name(), name))
    cdef int fd = shm_open(path, O_RDWR, mode0644)
    fstat(fd, &shmstat)
    cdef size_t new_size = shmstat.st_size * 2
    if ftruncate(fd, new_size) < 0:
        close(fd)
        raise Exception('ftruncate failed: %s' % errno)

    # update size field
    cdef Layout* layout = <Layout*>mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if <long>layout == -1:
        raise Exception('mmap failed: %s' % errno)
    layout.size = new_size

    # cleanup
    close(fd)
    munmap(layout, new_size)

'''
Get or create layout. If layout does not exist, setup it up.
'''
cdef Layout* get_layout(bytes name, size_t* output_size):
    cdef stat st
    cdef bytes path = bytes("sharedict_%s_%s" % (job_name(), name))
    cdef int fd = shm_open(path, O_CREAT | O_RDWR | O_EXCL, mode0644)
    cdef bool should_init = True
    cdef size_t size
    if fd == -1 and errno == EEXIST:
        should_init = False
        # create shm if not found
        fd = shm_open(path, O_RDWR, mode0644)
    if should_init:
        size = default_size
        if ftruncate(fd, default_size) < 0:
            close(fd)
            raise Exception('ftruncate failed: %s' % errno)
    else:
        should_sleep = False
        size = 0
        while size == 0:
            fstat(fd, &st)
            size = st.st_size
            if should_sleep:
                time.sleep(0.01)
            should_sleep = True

    cdef Layout* layout = <Layout*>mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if <long>layout == -1:
        raise Exception('mmap failed: %s' % errno)
    if should_init:
        # leader_pid will set with CAS later, instead here
        layout.size = size
        layout.state = STATE_INIT
    close(fd)
    output_size[0] = size
    return layout

cdef job_name():
    job_id = os.getenv("mapred_job_id")
    if job_id:
        return job_id
    else:
        return "test"

cdef class Sharedict:
    cdef Layout* _layout
    cdef char* _layout_name
    cdef float _load_factor
    cdef object _hash
    cdef size_t _mmap_size
    cdef object _loader

    def __init__(self, name, loader, hash = mmh3.hash, load_factor = 0.8):
        self._layout = get_layout(name, &self._mmap_size)
        self._layout_name = name
        self._load_factor = load_factor
        self._hash = hash
        self._loader = loader
        self.wait()

    def __dealloc__(self):
        munmap(self._layout, self._mmap_size)

    cdef _maintain(self):
        if self._try_acquire_leader():
            # if acquired leader, take responsibility to fill
            if self._layout.state != STATE_READY:
                self._fill(self._loader())

    cdef _remap(self):
        orig_size = self._mmap_size
        new_layout = get_layout(self._layout_name, &self._mmap_size)
        munmap(self._layout, orig_size)
        self._layout = new_layout

    cdef _try_acquire_leader(self):
        cdef int leader_pid = self._layout.leader_pid
        cdef int my_pid = os.getpid()
        if leader_pid == my_pid:
            # already acquired leader
            return True
        if leader_pid == 0 or not psutil.pid_exists(leader_pid):
            # leader not yet elected or died, abandon leader
            # exactly only one follower will succeed
            if cas_int(&self._layout.leader_pid, leader_pid, my_pid):
                return True
        return False

    cdef _ensure_size(self, size_t required_size):
        while required_size >= self._layout.size:
            extend_layout(self._layout_name)
            self._remap()

    '''
    Read stored blob at given position.
    Each blob is stored with format [length] [data], where length is 32-bit.
    '''
    cdef _read_blob(self, size_t pos):
        cdef char* layout_mem = <char*>self._layout
        cdef int* length_mem = <int*>(layout_mem + pos)
        return layout_mem[pos+4:pos+4+length_mem[0]]

    cdef _write_blob(self, bytes v):
        self._ensure_size(self._layout.blob_end + len(v) + 4)

        cdef size_t write_pos = self._layout.blob_end
        cdef char* layout_mem = <char*>self._layout
        cdef int* length_mem = <int*>(layout_mem + write_pos)

        length_mem[0] = len(v)

        cdef char* value_mem = v
        memcpy(&layout_mem[write_pos+4], value_mem, length_mem[0])
        self._layout.blob_end += length_mem[0] + 4
        return write_pos

    '''
    Compare stored blob at given position with given bytes (k).
    Returns True if different.
    '''
    cdef _compare_blob(self, int pos, bytes v):
        cdef char* value_mem = v
        cdef char* layout_mem = <char*>self._layout
        cdef int* length_mem = <int*>(layout_mem + pos)
        if length_mem[0] != len(v):
            return True
        if memcmp(&layout_mem[pos+4], value_mem, length_mem[0]) != 0:
            return True
        return False

    cdef _locate(self, bytes k, bool allow_empty):
        cdef HashEntry * entries = &self._layout.first_entry
        cdef int h = abs(self._hash(k)) % self._layout.num_entries
        while entries[h].key_pos > 0 and self._compare_blob(entries[h].key_pos, k):
            h += 37
            h %= self._layout.num_entries
        if entries[h].key_pos == 0 and not allow_empty:
            return -1
        return h

    cdef _fill(self, dic):
        cdef int num_elements = len(dic)
        cdef bytes ser_k
        cdef bytes ser_p
        cdef int entry
        self._layout.state = STATE_FILL
        self._layout.num_elements = num_elements 
        self._layout.num_entries = int(num_elements / self._load_factor)
        self._layout.blob_start = self._layout.num_entries * sizeof(HashEntry) + header_size
        self._layout.blob_end = self._layout.blob_start
        self._ensure_size(self._layout.blob_start)
        memset(&self._layout.first_entry, 0, self._layout.num_entries * sizeof(HashEntry))
        for k, v in dic.items():
            ser_k = marshal.dumps(k)
            ser_p = marshal.dumps(v)
            entry = self._locate(ser_k, True)
            (&self._layout.first_entry)[entry].key_pos = self._write_blob(ser_k)
            (&self._layout.first_entry)[entry].value_pos = self._write_blob(ser_p)
        self._layout.state = STATE_READY

    cdef _check_sanity(self):
        if self._layout.state != STATE_READY:
            raise Exception("Not ready")
        if self._layout.size != self._mmap_size:
            self._remap()

    '''
    Wait until ready.
    '''
    def wait(self):
        while self._layout.state != STATE_READY:
            self._maintain()
            time.sleep(0.1)

    def __getitem__(self, object key):
        self._check_sanity()
        cdef HashEntry *entries = &self._layout.first_entry
        cdef bytes ser_k = marshal.dumps(key)
        cdef int entry = self._locate(ser_k, False)
        if entry < 0:
            raise KeyError(key)
        cdef bytes s = self._read_blob(entries[entry].value_pos)
        return marshal.loads(s)

    def __len__(self):
        self._check_sanity()
        return self._layout.num_elements

    def __iter__(self):
        cdef bytes s
        self._check_sanity()
        for i in range(0, self._layout.num_entries):
            pos = (&self._layout.first_entry)[i].key_pos
            if pos > 0:
                s = self._read_blob(pos)
                yield marshal.loads(s)