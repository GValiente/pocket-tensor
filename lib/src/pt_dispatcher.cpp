/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_dispatcher.h"

#include <algorithm>
#include "pt_tweakme.h"
#include "pt_assert.h"

namespace pt
{

Dispatcher::Dispatcher() :
    Dispatcher(std::thread::hardware_concurrency())
{
}

Dispatcher::Dispatcher(std::size_t threadsCount) :
    _threadsCount(std::min(threadsCount, std::size_t(PT_MAX_CPU_THREADS))),
    _exit(false),
    _pendingTasks(0)
{
    PT_ASSERT(threadsCount > 0);

    if(threadsCount == 1)
    {
        return;
    }

    _threads.reserve(threadsCount);

    for(std::size_t index = 0; index < threadsCount; ++index)
    {
        _threads.emplace_back([this]
        {
            while(true)
            {
                std::unique_lock<std::mutex> lock(_mutex);

                while(! _exit && _tasks.empty())
                {
                    _condition.wait(lock);
                }

                if(_tasks.empty())
                {
                    return;
                }

                Task task = std::move(_tasks[0]);
                _tasks.pop_front();
                lock.unlock();
                task();

                std::unique_lock<std::mutex> pendingTasksLock(_pendingTasksMutex);

                --_pendingTasks;

                if(! _pendingTasks)
                {
                    pendingTasksLock.unlock();
                    _pendingTasksCondition.notify_one();
                }
            }
        });
    }
}

Dispatcher::~Dispatcher()
{
    if(_threadsCount == 1)
    {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(_mutex);

        _exit = true;
        _condition.notify_all();
    }

    for(auto& thread : _threads)
    {
        thread.join();
    }
}

void Dispatcher::add(Task&& task)
{
    if(_threadsCount == 1)
    {
        task();
        return;
    }

    std::unique_lock<std::mutex> lock(_mutex);

    _tasks.emplace_back(std::move(task));
    _condition.notify_one();

    std::unique_lock<std::mutex> pendingTasksLock(_pendingTasksMutex);

    ++_pendingTasks;
}

std::size_t Dispatcher::pendingTasks() noexcept
{
    if(_threadsCount == 1)
    {
        return 0;
    }

    std::unique_lock<std::mutex> pendingTasksLock(_pendingTasksMutex);

    return _pendingTasks;
}

void Dispatcher::join()
{
    if(_threadsCount == 1)
    {
        return;
    }

    std::unique_lock<std::mutex> pendingTasksLock(_pendingTasksMutex);

    _pendingTasksCondition.wait(pendingTasksLock, [this]{ return ! _pendingTasks; });
}

}
