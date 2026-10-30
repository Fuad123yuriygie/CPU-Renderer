// Minimal standalone probe used before the renderer existed: verify that the
// compiler's C++/thread runtime links statically and launches from a clean PATH.
#include <iostream>
#include <thread>
int main() {
    int answer=0;
    std::thread worker([&] { answer=42; });
    // join() both waits for the worker and makes its write visible to this thread.
    worker.join();
    std::cout << "Static C++20/thread runtime: " << answer << '\n';
    // A conventional process exit code makes the smoke test scriptable.
    return answer==42 ? 0 : 1;
}
