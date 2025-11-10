#include <vector>
#include <thread>
#include <queue>
#include <future>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : stop_flag(false)
    {
        for (size_t i = 0; i < num_threads; ++i) {
            // worker thread executes lambda
            workers.emplace_back([this] {
                while (true) {

                    // task callable initialization
                    std::function<void()> task;
                    
                    { // lock acquisition scope
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { 
                            return stop_flag || !tasks.empty(); 
                        });

                        // stop cmd + empty queue = worker exits
                        if (stop_flag && tasks.empty())
                            return;

                        // grab next task and set it for callable
                        task = std::move(tasks.front());
                        tasks.pop(); // TODO: check this
                    } // end lock scope

                    // execute the task
                    task();
                }
            });
        }
    }

    // Submit a callable and get a future to its result
    // NOTE: auto type, with -> to allow compiler to deduce return from F's return type
    // flipping order invalid bc F not defined yet
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>
    {
        // deduced return type to simplify future declaration below
        using return_type = typename std::invoke_result_t<F, Args...>;

        // like python's partial
        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        // get future before moving task into queue
        std::future<return_type> future = task_ptr->get_future();
        
        { // lock acquisition scope
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (stop_flag)
                throw std::runtime_error("ThreadPool has been stopped");

            tasks.emplace([task_ptr]() { (*task_ptr)(); });
        } // end lock scope

        // notify one waiting worker
        condition.notify_one();
        return future; // TODO: check we can recover results from caller thread?
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        condition.notify_all();

        for (std::thread &worker : workers)
            if (worker.joinable())
                worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;
};
