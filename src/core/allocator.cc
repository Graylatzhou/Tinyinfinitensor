#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        for (auto it = this->free_blocks.begin(); it != this->free_blocks.end(); it++)
        {
            if (it->second >= size)
            {
                size_t addr = it->first;
                size_t unused_mem = it->second - size;
                this->free_blocks.erase(it);
                used += size;
                if (used > peak)
                {
                    peak = used;
                }
                if (unused_mem > 0)
                {
                    this->free_blocks[addr + size] = unused_mem;
                }
                return addr;
            }
        }
        //如果没有合适的空闲块，直接分配新的内存，即内存已经分配满了
        size_t addr = used;
        used += size;
        if (used > peak)
        {
            peak = used;
        }
        return addr;
    }
// first - > addr second -> size
    void Allocator::free(size_t addr, size_t size)
    {
        //it->first = addr, it->second = size
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        //如果释放的内存块正好是最后一个内存块
        if (addr + size == peak){
            used -= size;
            peak -= size;
            return;
        }
        for (auto it = this->free_blocks.begin(); it != this->free_blocks.end(); it++){
            //释放的内存块在当前内存块后方，将这两个内存块合并
            if (it->first + it->second == addr){
                it->second += size;
                return;
            }
            // 此时释放的内存块位于it内存块的前方
            if (it->first == addr + size){
                free_blocks[addr] = size + it->second;
                free_blocks.erase(it);
                return;
            }
        }
        free_blocks[addr] = size;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
