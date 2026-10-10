#include "client.hpp"

#include <algorithm>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

#ifndef LONGBOOK27_VERSION
#define LONGBOOK27_VERSION "0.1.0"
#endif
#ifndef LONGBOOK27_ENGINE_COMMIT
#define LONGBOOK27_ENGINE_COMMIT "unknown"
#endif

namespace longbook {
volatile std::sig_atomic_t received_signal = 0;

namespace {
template <typename T>
T integer(const std::string & name, const std::string & value, T low, T high) {
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < low || result > high) {
        throw std::invalid_argument(name + ": invalid integer '" + value + "'");
    }
    return result;
}

float real(const std::string & name, const std::string & value, float low, float high) {
    char * end = nullptr;
    const float result = std::strtof(value.c_str(), &end);
    if (value.empty() || end != value.c_str() + value.size() || !std::isfinite(result) || result < low || result > high) {
        throw std::invalid_argument(name + ": invalid finite number '" + value + "'");
    }
    return result;
}
}

Options parse_options(int argc, char ** argv) {
    Options o;
    o.threads = static_cast<int>(std::max(1U, std::min(8U, std::thread::hardware_concurrency())));
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value");
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") o.help = true;
        else if (arg == "--version") o.version = true;
        else if (arg == "--model" || arg == "-m") o.model = value();
        else if (arg == "--prompt" || arg == "-p") { o.prompt = value(); o.prompt_given = true; }
        else if (arg == "--prompt-file" || arg == "-f") o.prompt_file = value();
        else if (arg == "--system") o.system = value();
        else if (arg == "--ctx-size" || arg == "-c") o.context = integer<int>(arg, value(), 32, 262144);
        else if (arg == "--predict" || arg == "-n") o.predict = integer<int>(arg, value(), 1, 262144);
        else if (arg == "--threads" || arg == "-t") o.threads = integer<int>(arg, value(), 1, 1024);
        else if (arg == "--gpu-layers" || arg == "-ngl") o.gpu_layers = integer<int>(arg, value(), -1, 1024);
        else if (arg == "--batch-size" || arg == "-b") o.batch = integer<int>(arg, value(), 1, 262144);
        else if (arg == "--ubatch-size" || arg == "-ub") o.ubatch = integer<int>(arg, value(), 1, 262144);
        else if (arg == "--top-k") o.top_k = integer<int>(arg, value(), 0, 1000000);
        else if (arg == "--seed") o.seed = integer<std::uint32_t>(arg, value(), 0U, std::numeric_limits<std::uint32_t>::max());
        else if (arg == "--temp") o.temperature = real(arg, value(), 0.0F, 10.0F);
        else if (arg == "--top-p") o.top_p = real(arg, value(), 0.000001F, 1.0F);
        else if (arg == "--thinking") {
            const auto v = value();
            if (v != "on" && v != "off") throw std::invalid_argument("--thinking requires on or off");
            o.thinking = v == "on";
        }
        else if (arg == "--interactive" || arg == "-i") o.interactive = true;
        else if (arg == "--raw") o.raw = true;
        else if (arg == "--inspect") o.inspect = true;
        else if (arg == "--allow-compatible-model") o.compatible = true;
        else if (arg == "--verbose") o.verbose = true;
        else if (arg == "--image" || arg == "--mmproj") throw std::invalid_argument("image input is not supported: the pinned GGUF repository has no compatible mmproj");
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.help || o.version) return o;
    if (o.model.empty()) throw std::invalid_argument("--model is required; use --help for examples");
    if (o.inspect) return o;
    const int sources = static_cast<int>(o.prompt_given) + static_cast<int>(!o.prompt_file.empty()) + static_cast<int>(o.interactive);
    if (sources != 1) throw std::invalid_argument("choose exactly one of --prompt, --prompt-file, or --interactive");
    if (o.raw && (o.interactive || !o.system.empty() || o.thinking)) throw std::invalid_argument("--raw is single-prompt mode; omit --interactive, --system and --thinking on");
    if (o.predict >= o.context) throw std::invalid_argument("context budget exceeded: --predict must leave room for the prompt");
    o.batch = std::min(o.batch, o.context);
    o.ubatch = std::min(o.ubatch, o.batch);
    return o;
}

void print_help(std::ostream & out) {
    out << R"(Longbook27 — local Qwen35 GDN64 GGUF client

Usage:
  longbook27 -m MODEL.gguf -p "Explain this idea."
  longbook27 -m MODEL.gguf -f book.txt -c 32768 -n 1024
  longbook27 -m MODEL.gguf -i --gpu-layers 99
  longbook27 -m MODEL.gguf --inspect

Inputs:
  -m, --model PATH          Local GGUF file (never downloaded implicitly)
  -p, --prompt TEXT         One text prompt
  -f, --prompt-file PATH    UTF-8 text file; '-' reads stdin (64 MiB limit)
  -i, --interactive         Line-oriented chat; /reset and /exit
      --system TEXT        Optional system message
      --thinking on|off    Default off; on uses medium reasoning effort
      --raw                Tokenize the input directly, without chat formatting
      --inspect            JSON header/topology inspection; no weight loading

Inference:
  -c, --ctx-size N          Token budget, 32..262144 (default 8192)
  -n, --predict N           Maximum new tokens (default 512)
  -t, --threads N           CPU threads (default min(8, hardware concurrency))
  -ngl, --gpu-layers N      GPU layers (default 0; -1 or 99 for all)
  -b, --batch-size N        Logical prefill batch (default 512)
  -ub, --ubatch-size N      Physical batch (default 128; capped to logical batch)
      --temp F             Temperature (default 1; 0 = greedy)
      --top-k N            Top-k (default 20; 0 disables)
      --top-p F            Nucleus probability (default .95)
      --seed N             Sampling seed (default 42)
      --allow-compatible-model
                           Explicitly permit other all-recurrent qwen35 shapes
                           for engine development; not model identity verification
      --verbose            Include engine info/debug logs on stderr
  -h, --help               Show this help
      --version            Show client and pinned engine versions

Output text is streamed to stdout; prompts, diagnostics, and timing go to stderr.
Chat re-evaluates complete history on each turn. It never silently truncates or
shifts recurrent state. Context overflow is an error; /reset starts a new session.
Ctrl-C stops inference (GPU work may finish the current batch first).
This build provides text input. It does not include an HTTP server or image input.
)";
}

std::string read_prompt(const std::string & path, std::istream & input) {
    std::ifstream file;
    std::istream * stream = &input;
    if (path != "-") {
        file.open(path, std::ios::binary);
        if (!file) throw std::invalid_argument("cannot open prompt file: " + path);
        stream = &file;
    }
    std::string text;
    char buffer[16384];
    while (stream->read(buffer, sizeof(buffer)) || stream->gcount() > 0) {
        const auto count = static_cast<std::size_t>(stream->gcount());
        if (count > max_input_bytes - text.size()) throw std::invalid_argument("prompt exceeds 64 MiB input limit");
        text.append(buffer, count);
    }
    if (stream->bad()) throw std::runtime_error("failed while reading prompt file");
    if (text.find('\0') != std::string::npos) throw std::invalid_argument("prompt contains NUL bytes; provide UTF-8 text");
    return text;
}
} // namespace longbook

namespace {
void stop_handler(int signal) { longbook::received_signal = signal; }

void report(const longbook::Result & r) {
    std::cerr << "\n[longbook27] prompt_tokens=" << r.prompt_tokens
              << " generated_tokens=" << r.generated_tokens << " stop=" << r.stop
              << " prefill_ms=" << r.prefill_ms << " generation_ms=" << r.generation_ms << '\n';
}
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
#if defined(_WIN32)
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
#else
    // Interrupt idle stdin reads as well as inference; SA_RESTART would leave
    // an interactive process waiting for another line after SIGTERM.
    struct sigaction action{};
    action.sa_handler = stop_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0) {
        std::cerr << "[longbook27] failed to install signal handlers\n";
        return 1;
    }
#endif
    try {
        auto options = longbook::parse_options(argc, argv);
        if (options.help) { longbook::print_help(std::cout); return 0; }
        if (options.version) {
            std::cout << "Longbook27 " << LONGBOOK27_VERSION << "\nengine " << LONGBOOK27_ENGINE_COMMIT << '\n';
            return 0;
        }
        if (options.inspect) { longbook::inspect_model(options, std::cout); return 0; }
        if (!options.prompt_file.empty()) options.prompt = longbook::read_prompt(options.prompt_file, std::cin);
        if (options.prompt.size() > longbook::max_input_bytes || options.system.size() > longbook::max_input_bytes) {
            throw std::invalid_argument("input exceeds 64 MiB limit");
        }
        if (!options.interactive && options.prompt.empty()) throw std::invalid_argument("prompt must not be empty");
        longbook::Client client(options);
        std::vector<longbook::Message> history;
        if (!options.system.empty()) history.push_back({"system", options.system, {}});
        if (!options.interactive) {
            history.push_back({"user", options.prompt, {}});
            const auto result = client.generate(history, std::cout);
            std::cout << '\n';
            report(result);
            return longbook::received_signal ? 128 + longbook::received_signal : 0;
        }
        std::cerr << "Longbook27 chat. /reset clears history; /exit quits.\n";
        for (;;) {
            std::cerr << "you> " << std::flush;
            std::string line;
            if (!std::getline(std::cin, line)) break;
            if (line == "/exit" || line == "/quit") break;
            if (line == "/reset") {
                history.clear();
                if (!options.system.empty()) history.push_back({"system", options.system, {}});
                std::cerr << "[longbook27] history reset\n";
                continue;
            }
            if (line.empty()) continue;
            longbook::received_signal = 0;
            history.push_back({"user", line, {}});
            try {
                const auto result = client.generate(history, std::cout);
                std::cout << '\n';
                report(result);
                if (longbook::received_signal) {
                    history.pop_back();
                    const auto signal = longbook::received_signal;
                    longbook::received_signal = 0;
                    if (signal == SIGTERM) return 128 + SIGTERM;
                    std::cerr << "[longbook27] interrupted turn discarded\n";
                } else history.push_back(longbook::assistant_message(result.text, options.thinking));
            } catch (const longbook::Interrupted &) {
                history.pop_back();
                const auto signal = longbook::received_signal;
                longbook::received_signal = 0;
                if (signal == SIGTERM) return 128 + SIGTERM;
                std::cerr << "\n[longbook27] interrupted turn discarded\n";
            } catch (const std::invalid_argument & e) {
                history.pop_back();
                std::cerr << "[longbook27] " << e.what() << "\nUse /reset or a shorter prompt.\n";
            }
        }
        return longbook::received_signal ? 128 + longbook::received_signal : 0;
    } catch (const longbook::Interrupted &) {
        std::cerr << "\n[longbook27] interrupted\n";
        return 128 + (longbook::received_signal ? longbook::received_signal : SIGINT);
    } catch (const std::invalid_argument & e) {
        std::cerr << "[longbook27] " << e.what() << '\n';
        return 2;
    } catch (const std::exception & e) {
        std::cerr << "[longbook27] " << e.what() << '\n';
        return longbook::received_signal ? 128 + longbook::received_signal : 1;
    }
}
