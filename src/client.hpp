#pragma once

#include <csignal>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace longbook {

constexpr std::size_t max_input_bytes = 64U * 1024U * 1024U;

struct Options {
    std::string model;
    std::string prompt;
    std::string prompt_file;
    std::string system;
    int context = 8192;
    int predict = 512;
    int threads = 4;
    int gpu_layers = 0;
    int batch = 512;
    int ubatch = 128;
    int top_k = 20;
    std::uint32_t seed = 42;
    float temperature = 1.0F;
    float top_p = 0.95F;
    bool prompt_given = false;
    bool interactive = false;
    bool raw = false;
    bool thinking = false;
    bool inspect = false;
    bool compatible = false;
    bool verbose = false;
    bool help = false;
    bool version = false;
};

struct Message {
    std::string role;
    std::string content;
    std::string reasoning;
};

struct Result {
    std::string text;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    std::string stop = "limit";
    double prefill_ms = 0;
    double generation_ms = 0;
};

class Interrupted : public std::runtime_error {
public:
    Interrupted() : std::runtime_error("inference interrupted") {}
};

extern volatile std::sig_atomic_t received_signal;

Options parse_options(int argc, char ** argv);
void print_help(std::ostream & out);
std::string read_prompt(const std::string & path, std::istream & input);
std::string render_chat(const std::vector<Message> & messages, bool thinking);
Message assistant_message(const std::string & generated, bool thinking);
void inspect_model(const Options & options, std::ostream & out);

class Client {
public:
    explicit Client(const Options & options);
    ~Client();
    Client(const Client &) = delete;
    Client & operator=(const Client &) = delete;
    Result generate(const std::vector<Message> & messages, std::ostream & output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace longbook
