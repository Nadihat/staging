#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <charconv>
#include <csignal>
#include <format>
#include <cstdlib>

namespace fs = std::filesystem;

// Global atomic to ensure the compiler does not optimize away the CPU-heavy loop
std::atomic<uint32_t> g_last_hash{0};

// --- Helper Functions ---

// Strip leading and trailing whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Portable validation of a positive integer matching argparse's behavior
int parse_positive_int(const std::string& arg_name, const std::string& value) {
    try {
        size_t pos;
        long long ivalue = std::stoll(value, &pos);
        if (pos != value.size()) {
            std::cerr << "Error: argument " << arg_name << ": Invalid int value: '" << value << "'\n";
            std::exit(1);
        }
        if (ivalue <= 0) {
            std::cerr << "Error: argument " << arg_name << ": Invalid positive int value: '" << value << "'\n";
            std::exit(1);
        }
        return static_cast<int>(ivalue);
    } catch (...) {
        std::cerr << "Error: argument " << arg_name << ": Invalid int value: '" << value << "'\n";
        std::exit(1);
    }
}

// Display command help
void print_help() {
    std::cout << "Usage: NewHash [options]\n\n"
              << "A simple script to repeat intentions hourly from a file or a direct input.\n\n"
              << "Options:\n"
              << "  --file FILE         The file containing the intention.\n"
              << "  --intent INTENT     The intention string.\n"
              << "  --repeats REPEATS   Number of times to repeat the intention.\n"
              << "  --duration DURATION Duration in seconds to repeat the intentions for.\n"
              << "  --help, -?          Show this help message and exit.\n";
}

// Read intention from file safely matching Python's UTF-8 fallback
std::string read_intention_from_file(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file descriptor");
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        return trim(content);
    } catch (const std::exception& e) {
        std::cout << "Error reading file '" << filename << "': " << e.what() << "\n";
        std::exit(1);
    }
}

// Format time as HH:MM:SS using C++20 std::format [1]
std::string format_time(int seconds) {
    int hrs = seconds / 3600;
    int mins = (seconds % 3600) / 60;
    int secs = seconds % 60;
    return std::format("{:02}:{:02}:{:02}", hrs, mins, secs);
}

// Prompt intention fallback loop
std::pair<std::string, std::string> prompt_intention_or_file() {
    while (true) {
        std::cout << "Intention (or filename): ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            std::cout << "\n";
            std::exit(0);
        }
        user_input = trim(user_input);
        if (user_input.empty()) {
            std::cout << "Input cannot be empty. Please enter a valid intention or filename.\n";
            continue;
        }
        if (fs::is_regular_file(user_input)) {
            return {user_input, ""}; // First element is file, second is intent
        } else {
            return {"", user_input};
        }
    }
}

// Prompt positive integer fallback loop
int prompt_positive_int(const std::string& prompt_text) {
    while (true) {
        std::cout << prompt_text;
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            std::cout << "\n";
            std::exit(0);
        }
        user_input = trim(user_input);
        try {
            size_t pos;
            long long value = std::stoll(user_input, &pos);
            if (pos != user_input.size() || value <= 0) {
                std::cout << "Please enter a positive integer.\n";
                continue;
            }
            return static_cast<int>(value);
        } catch (...) {
            std::cout << "Invalid input. Please enter a positive integer.\n";
        }
    }
}

// Background task performing the heavy polynomial rolling hash
void execute_intentions(std::string base_intent, int repeats) {
    char num_buf[12];
    for (int r = 0; r < repeats; ++r) {
        std::string current_str = base_intent;
        uint32_t h = 0;

        for (int iter = 0; iter < 88888888; ++iter) {
            h = 0;
            // Native execution of (h * 31 + ord(char)) & 0xFFFFFFFF
            for (char c : current_str) {
                h = h * 31 + static_cast<uint8_t>(c);
            }
            // Fast C++20 single-pass integer-to-string conversion [1]
            auto [ptr, ec] = std::to_chars(num_buf, num_buf + 11, h);
            current_str = std::string(num_buf, ptr - num_buf);
        }

        // Store the final loop state in atomic variable to enforce CPU utilization
        g_last_hash.store(h, std::memory_order_relaxed);

        #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : : "memory"); // Compiler memory barrier guard
        #endif
    }
}

// Portable signal handler for SIGINT (Ctrl+C)
void sigint_handler(int sig) {
    (void)sig;
    std::cout << "\n\nScript interrupted by user. Exiting gracefully.\n";
    std::exit(0);
}

// Wall clock helper matching Python's time.time()
double get_current_time_seconds() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

// --- Program Entry Point ---
int main(int argc, char* argv[]) {
    std::signal(SIGINT, sigint_handler);

    std::cout << "ServitorConnect CLI v3\n";
    std::cout << "by AnthroHeart/Anthro Teacher/Thomas Sweet\n\n";

    std::string file_path = "";
    std::string intent_str = "";
    int repeats = -1;
    int duration = -1;

    // Strict manual replication of Python's Argparse behavior
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-?") {
            print_help();
            std::exit(0);
        } else if (arg == "--file") {
            if (i + 1 >= argc) {
                std::cerr << "Error: argument --file: expected one argument\n";
                std::exit(1);
            }
            if (!intent_str.empty()) {
                std::cerr << "Error: argument --file: not allowed with argument --intent\n";
                std::exit(1);
            }
            file_path = argv[++i];
        } else if (arg == "--intent") {
            if (i + 1 >= argc) {
                std::cerr << "Error: argument --intent: expected one argument\n";
                std::exit(1);
            }
            if (!file_path.empty()) {
                std::cerr << "Error: argument --intent: not allowed with argument --file\n";
                std::exit(1);
            }
            intent_str = argv[++i];
        } else if (arg == "--repeats") {
            if (i + 1 >= argc) {
                std::cerr << "Error: argument --repeats: expected one argument\n";
                std::exit(1);
            }
            repeats = parse_positive_int("--repeats", argv[++i]);
        } else if (arg == "--duration") {
            if (i + 1 >= argc) {
                std::cerr << "Error: argument --duration: expected one argument\n";
                std::exit(1);
            }
            duration = parse_positive_int("--duration", argv[++i]);
        } else {
            std::cerr << "Error: unrecognized argument: " << arg << "\n";
            print_help();
            std::exit(1);
        }
    }

    // Handle Intention or File Fallback Prompts
    if (file_path.empty() && intent_str.empty()) {
        auto prompt_res = prompt_intention_or_file();
        file_path = prompt_res.first;
        intent_str = prompt_res.second;
    }

    std::string file_contents = "";
    std::string source = "";

    if (!file_path.empty()) {
        if (!fs::is_regular_file(file_path)) {
            std::cout << "File '" << file_path << "' does not exist. Exiting.\n";
            std::exit(1);
        }
        source = "File '" + file_path + "'";
        file_contents = read_intention_from_file(file_path);
    } else {
        source = "Intent '" + intent_str + "'";
    }

    // Handle Repeats Fallback Prompt
    if (repeats == -1) {
        repeats = prompt_positive_int("Number of repeats per hour: ");
    }

    // Handle Duration Fallback Prompt
    if (duration == -1) {
        duration = prompt_positive_int("Duration in seconds: ");
    }

    // Display configuration state
    std::cout << "\nSource: " << source << "\n";
    std::cout << "Repeats per Hour: " << repeats << "\n";
    std::cout << "Duration: " << format_time(duration) << "\n";

    double start_time = get_current_time_seconds();
    double end_time = start_time + duration;
    double next_hour_time = start_time + 3600.0;

    std::string base_intent = (!file_path.empty()) ? file_contents : intent_str;

    std::cout << "\nRepeating Intention...\n";
    
    // Spawn background daemon-like thread matching Python's threading.Thread(daemon=True)
    std::thread(execute_intentions, base_intent, repeats).detach();

    while (true) {
        double current_time = get_current_time_seconds();

        if (current_time >= end_time) {
            break;
        }

        if (current_time >= next_hour_time) {
            std::cout << "\nRepeating Intention...\n";
            std::thread(execute_intentions, base_intent, repeats).detach();
            next_hour_time += 3600.0;
        }

        int remaining = static_cast<int>(end_time - current_time);
        if (remaining < 0) remaining = 0;

        std::string timer_str = format_time(remaining);
        std::string output = source + " Repeated " + std::to_string(repeats) + " Times Hourly: " + timer_str;

        std::cout << "\r" << output << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\nDuration completed.\n";
    return 0;
}
