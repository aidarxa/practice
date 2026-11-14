/** @file @brief CPU-only behavioral checks for the bounded diagnostic event recorder. */
#include "probe_log.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
}

int main() {
    try {
        db::s2probe::EventLog log(2);
        log.add({"sample", 10, 20, 1, 42, -7, "quote\"\n\\", {"arg", "two words"}});
        log.add({"nested", 11, 19, 1, 0, 0, "", {}});
        log.add({"discarded", 21, 22, 2, 0, 0, "", {}});
        std::ostringstream out;
        log.write(out);
        const auto text = out.str();
        require(text.find("\"dropped\":1") != std::string::npos, "capacity loss must be explicit");
        require(text.find("discarded") == std::string::npos, "capacity must be bounded");
        require(text.find("quote\\\"\\u000a\\\\") != std::string::npos, "JSON escaping must be lossless");
        require(text.find("\"status\":-7") != std::string::npos, "signed native status must survive");
        require(text.find("\"bytes\":42") != std::string::npos, "request size must survive");
        require(text.find("two words") != std::string::npos, "argv must survive");
        require(log.nextCompilation() == 1 && log.nextCompilation() == 2, "compilation IDs must be unique");
        db::s2probe::EventLog empty(0);
        empty.add({"never_stored"});
        std::ostringstream empty_out;
        empty.write(empty_out);
        require(empty_out.str().find("\"event_count\":0,\"dropped\":1") != std::string::npos, "zero capacity must not store an event");
        db::s2probe::EventLog concurrent(400);
        std::vector<std::thread> threads;
        for (int i=0;i<4;++i) threads.emplace_back([&] {
            for (int j=0;j<100;++j) concurrent.add({"parallel", 1, 2, 0, 0, 0, "", {}});
        });
        for (auto& thread:threads) thread.join();
        std::ostringstream serialized;
        concurrent.write(serialized);
        require(serialized.str().find("\"event_count\":400") != std::string::npos, "concurrent events must not disappear");
        require(serialized.str().find("\"dropped\":0") != std::string::npos, "capacity boundary is inclusive");
        const auto before=db::s2probe::nowNs();
        require(db::s2probe::nowNs() >= before, "timestamps must be monotonic");
        {
            db::s2probe::CompilationScope parent;
            db::s2probe::Span first("parent");
            {
                db::s2probe::CompilationScope child;
                db::s2probe::Span nested("child");
                nested.finish(0);
            }
            db::s2probe::Span restored("restored");
            restored.finish(-9);
            first.finish(0);
        }
        std::ostringstream global;
        db::s2probe::log().write(global);
        require(global.str().find("\"compilations\":2") != std::string::npos, "nested grouping needs independent IDs");
        require(global.str().find("\"kind\":\"parent\"") != std::string::npos, "finished direct spans must be exported");
        const auto restored=global.str().substr(global.str().find("\"kind\":\"restored\""));
        require(restored.substr(0,restored.find('}')).find("\"compilation\":1") != std::string::npos, "child scope must restore its parent ID");
        require(restored.substr(0,restored.find('}')).find("\"status\":-9") != std::string::npos, "span status must survive");
        std::cout << text << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
