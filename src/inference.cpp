#include "../include/inference.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <new>
#include <unistd.h>

SharedArena* init_shared_memory() {
    // create POSIX Shared Memory
    int fd = shm_open("/inference_arena", O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedArena));

    // map into cpu address space
    void* ptr = mmap(0, sizeof(SharedArena), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    SharedArena* arena = static_cast<SharedArena*>(ptr);

    // cudaHostRegisterMapped maps CPU pointer to a GPU pointer
    cudaHostRegister(arena, sizeof(SharedArena), cudaHostRegisterMapped);

    new (&arena->curr_row_ptr) std::atomic<int>(0);
    new (&arena->active_batches) std::atomic<int>(0);
    
    return arena;
}