/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_DISPATCHER_H
#define PT_DISPATCHER_H

#include <deque>
#include <mutex>
#include <vector>
#include <thread>
#include <functional>
#include <condition_variable>

namespace pt
{

class Dispatcher
{

public:
    using Task = std::function<void(void)>;

    Dispatcher();

    explicit Dispatcher(std::size_t threads);

    ~Dispatcher();

    std::size_t threads() const noexcept
    {
        return _threadsCount;
    }

    void add(Task&& task);

    std::size_t pendingTasks() noexcept;

    void join();

protected:
    std::mutex _mutex;
    std::condition_variable _condition;
    std::deque<Task> _tasks;
    std::vector<std::thread> _threads;
    std::size_t _threadsCount;
    bool _exit;

    std::mutex _pendingTasksMutex;
    std::condition_variable _pendingTasksCondition;
    std::size_t _pendingTasks;
};

}

#endif
