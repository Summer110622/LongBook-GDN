#include "client.hpp"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

#ifndef LONGBOOK27_ENGINE_COMMIT
#define LONGBOOK27_ENGINE_COMMIT "unknown"
#endif

namespace longbook {
namespace {
using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
using Header = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

bool abort_inference(void *) { return received_signal != 0; }
bool load_progress(float, void *) { return received_signal == 0; }

void log_message(ggml_log_level level, const char * text, void * data) {
    const bool verbose = *static_cast<const bool *>(data);
    if (verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

std::string trim(const std::string & value) {
    const auto first = value.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n\f\v") - first + 1);
}

std::string json_string(const std::string & text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}

std::uint64_t number(const gguf_context * h, const char * key, std::uint64_t fallback = 0) {
    const auto id = gguf_find_key(h, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(h, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(h, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(h, id);
        case GGUF_TYPE_INT32: {
            const auto n = gguf_get_val_i32(h, id);
            return n >= 0 ? static_cast<std::uint64_t>(n) : fallback;
        }
        default: return fallback;
    }
}

std::string string_value(const gguf_context * h, const char * key) {
    const auto id = gguf_find_key(h, key);
    return id >= 0 && gguf_get_kv_type(h, id) == GGUF_TYPE_STRING ? gguf_get_val_str(h, id) : "";
}

struct ModelInfo {
    Header header{nullptr, gguf_free};
    std::string architecture;
    std::uint64_t file_bytes = 0;
    std::uint64_t blocks = 0;
    std::uint64_t mtp = 0;
    std::uint64_t recurrent = 0;
    std::uint64_t context = 0;
    std::uint64_t embedding = 0;
    bool complete = false;
    bool all_recurrent = false;
    bool signature = false;
};

struct BoundedHeaderReader {
    std::ifstream & input;
    std::uint64_t position = 0;
    static constexpr std::uint64_t limit = 32U * 1024U * 1024U;
};

std::size_t read_header(void * userdata, void * output, std::uint64_t offset, std::size_t bytes) {
    auto & reader = *static_cast<BoundedHeaderReader *>(userdata);
    if (offset > BoundedHeaderReader::limit || bytes > BoundedHeaderReader::limit - offset) return 0;
    if (reader.position != offset) {
        reader.input.clear();
        reader.input.seekg(static_cast<std::streamoff>(offset));
    }
    reader.input.read(static_cast<char *>(output), static_cast<std::streamsize>(bytes));
    const auto count = static_cast<std::size_t>(reader.input.gcount());
    reader.position = offset + count;
    return count;
}

ModelInfo model_info(const std::string & path) {
    if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("model is not a regular file: " + path);
    std::ifstream input(path, std::ios::binary);
    unsigned char prelude[24]{};
    input.read(reinterpret_cast<char *>(prelude), sizeof(prelude));
    if (!input || std::string(reinterpret_cast<char *>(prelude), 4) != "GGUF") throw std::invalid_argument("invalid GGUF header: " + path);
    const auto little_endian = [&](int start, int size) {
        std::uint64_t value = 0;
        for (int i = size - 1; i >= 0; --i) value = (value << 8U) | prelude[start + i];
        return value;
    };
    if (little_endian(4, 4) != 3) throw std::invalid_argument("Longbook27 requires little-endian GGUF version 3");
    if (little_endian(8, 8) == 0 || little_endian(8, 8) > 4096 || little_endian(16, 8) == 0 || little_endian(16, 8) > 256) {
        throw std::invalid_argument("GGUF header exceeds the supported tensor/metadata count");
    }
    ModelInfo info;
    info.file_bytes = std::filesystem::file_size(path);
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    BoundedHeaderReader reader{input, sizeof(prelude)};
    info.header.reset(gguf_init_from_callback(read_header, &reader, 1024 * 1024,
        std::min(info.file_bytes, BoundedHeaderReader::limit), params));
    if (!info.header) throw std::invalid_argument("cannot read GGUF metadata: " + path);
    const auto * h = info.header.get();
    if (gguf_get_version(h) != 3) throw std::invalid_argument("Longbook27 requires GGUF version 3");
    info.architecture = string_value(h, "general.architecture");
    info.blocks = number(h, "qwen35.block_count");
    info.mtp = number(h, "qwen35.nextn_predict_layers");
    info.context = number(h, "qwen35.context_length");
    info.embedding = number(h, "qwen35.embedding_length");
    const auto count = gguf_get_n_tensors(h);
    const std::uint64_t data_offset = gguf_get_data_offset(h);
    info.complete = count > 0 && data_offset <= info.file_bytes;
    for (std::int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(h, i);
        const std::string suffix = ".ssm_conv1d.weight";
        if (name.rfind("blk.", 0) == 0 && name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) ++info.recurrent;
        const std::uint64_t offset = gguf_get_tensor_offset(h, i);
        const std::uint64_t bytes = gguf_get_tensor_size(h, i);
        if (data_offset > info.file_bytes || offset > info.file_bytes - data_offset || bytes > info.file_bytes - data_offset - offset) info.complete = false;
    }
    const auto main_blocks = info.blocks > info.mtp ? info.blocks - info.mtp : 0;
    info.all_recurrent = info.architecture == "qwen35" && main_blocks > 0 && main_blocks <= 256 && info.recurrent == main_blocks;
    for (std::uint64_t i = 0; info.all_recurrent && i < main_blocks; ++i) {
        const auto block = "blk." + std::to_string(i);
        info.all_recurrent = gguf_find_tensor(h, (block + ".ssm_conv1d.weight").c_str()) >= 0
                          && gguf_find_tensor(h, (block + ".attn_q.weight").c_str()) < 0
                          && gguf_find_tensor(h, (block + ".attn_k.weight").c_str()) < 0;
    }
    info.signature = info.all_recurrent && info.blocks == 65 && info.mtp == 1 && info.recurrent == 64
        && info.context == 262144 && info.embedding == 5120 && count == 899
        && number(h, "general.file_type") == 15
        && number(h, "qwen35.feed_forward_length") == 17408
        && number(h, "qwen35.ssm.conv_kernel") == 4
        && number(h, "qwen35.ssm.state_size") == 128
        && number(h, "qwen35.ssm.group_count") == 16
        && number(h, "qwen35.ssm.time_step_rank") == 48
        && number(h, "qwen35.ssm.inner_size") == 6144;
    return info;
}

void validate_model(const ModelInfo & info, const Options & options) {
    if (!info.all_recurrent) throw std::invalid_argument("unsupported architecture/topology: an all-recurrent qwen35 GGUF is required");
    if (!options.compatible && !info.signature) throw std::invalid_argument("Longbook27 signature mismatch; use the pinned Q4_K_M checkpoint (or --allow-compatible-model for engine development)");
    if (!info.complete) throw std::invalid_argument("GGUF tensor data is truncated; complete the model download");
    if (info.context == 0 || static_cast<std::uint64_t>(options.context) > info.context) {
        throw std::invalid_argument("requested context exceeds model metadata context_length=" + std::to_string(info.context));
    }
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    if (text.empty()) throw std::invalid_argument("prompt must not be empty");
    if (text.size() > max_input_bytes || text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("input exceeds 64 MiB limit");
    }
    const auto bytes = static_cast<std::int32_t>(text.size());
    const auto needed = llama_tokenize(vocab, text.data(), bytes, nullptr, 0, true, special);
    if (needed == std::numeric_limits<std::int32_t>::min()) throw std::runtime_error("tokenizer size overflow");
    const auto capacity = needed < 0 ? -needed : needed;
    if (capacity <= 0) throw std::invalid_argument("prompt contains no tokens");
    std::vector<llama_token> tokens(static_cast<std::size_t>(capacity));
    const auto actual = llama_tokenize(vocab, text.data(), bytes, tokens.data(), capacity, true, special);
    if (actual <= 0 || actual > capacity) throw std::runtime_error("tokenization failed");
    tokens.resize(static_cast<std::size_t>(actual));
    return tokens;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char initial[256];
    auto length = llama_token_to_piece(vocab, token, initial, sizeof(initial), 0, true);
    if (length >= 0) return std::string(initial, static_cast<std::size_t>(length));
    if (length == std::numeric_limits<std::int32_t>::min()) throw std::runtime_error("token piece size overflow");
    std::string result(static_cast<std::size_t>(-length), '\0');
    length = llama_token_to_piece(vocab, token, result.data(), static_cast<std::int32_t>(result.size()), 0, true);
    if (length < 0) throw std::runtime_error("token detokenization failed");
    result.resize(static_cast<std::size_t>(length));
    return result;
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

std::string render_chat(const std::vector<Message> & messages, bool thinking) {
    // Audited text-only subset of the pinned model's template, reasoning_effort=medium.
    // Tools, multimodal content, and arbitrary Jinja execution are deliberately absent.
    std::string result;
    for (const auto & message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant") throw std::invalid_argument("unsupported chat role");
        if (message.content.size() > max_input_bytes || message.reasoning.size() > max_input_bytes) throw std::invalid_argument("chat exceeds 64 MiB input limit");
        result += "<|im_start|>" + message.role + "\n";
        if (message.role == "assistant") result += "<think>\n" + trim(message.reasoning) + "\n</think>\n\n";
        result += trim(message.content) + "<|im_end|>\n";
        if (result.size() > max_input_bytes) throw std::invalid_argument("chat exceeds 64 MiB input limit");
    }
    result += "<|im_start|>assistant\n<think>\n";
    if (!thinking) result += "\n</think>\n\n";
    return result;
}

Message assistant_message(const std::string & generated, bool thinking) {
    if (!thinking) return {"assistant", generated, {}};
    const auto end = generated.find("</think>");
    if (end == std::string::npos) return {"assistant", {}, trim(generated)};
    return {"assistant", trim(generated.substr(end + 8)), trim(generated.substr(0, end))};
}

void inspect_model(const Options & options, std::ostream & out) {
    const auto info = model_info(options.model);
    out << "{\n"
        << "  \"architecture\": " << json_string(info.architecture) << ",\n"
        << "  \"gguf_version\": " << gguf_get_version(info.header.get()) << ",\n"
        << "  \"tensor_count\": " << gguf_get_n_tensors(info.header.get()) << ",\n"
        << "  \"declared_blocks\": " << info.blocks << ",\n"
        << "  \"declared_mtp_blocks\": " << info.mtp << ",\n"
        << "  \"recurrent_blocks\": " << info.recurrent << ",\n"
        << "  \"context_length\": " << info.context << ",\n"
        << "  \"embedding_length\": " << info.embedding << ",\n"
        << "  \"file_size_bytes\": " << info.file_bytes << ",\n"
        << "  \"tensor_extents_fit_file\": " << (info.complete ? "true" : "false") << ",\n"
        << "  \"all_recurrent\": " << (info.all_recurrent ? "true" : "false") << ",\n"
        << "  \"longbook27_signature\": " << (info.signature ? "true" : "false") << ",\n"
        << "  \"weights_verified\": false,\n"
        << "  \"engine_commit\": " << json_string(LONGBOOK27_ENGINE_COMMIT) << "\n}\n";
}

struct Client::Impl {
    Options options;
    Model model{nullptr, llama_model_free};
    Context context{nullptr, llama_free};
    const llama_vocab * vocab = nullptr;

    explicit Impl(const Options & o) : options(o) {
        // The callback points into this object, so restore the global logger on teardown.
        llama_log_set(log_message, &options.verbose);
        try {
            const auto info = model_info(options.model);
            validate_model(info, options);
            if (options.compatible) std::cerr << "[longbook27] compatible-model mode: checkpoint identity is not verified\n";
            ggml_backend_load_all();
            llama_backend_init();
            if (options.gpu_layers != 0 && !llama_supports_gpu_offload()) throw std::invalid_argument("GPU offload requested but no GPU backend/device is available; use --gpu-layers 0 or a GPU build");
            auto parameters = llama_model_default_params();
            parameters.n_gpu_layers = options.gpu_layers;
            parameters.load_mode = LLAMA_LOAD_MODE_MMAP;
            parameters.load_mtp = false;
            parameters.progress_callback = load_progress;
            model.reset(llama_model_load_from_file(options.model.c_str(), parameters));
            if (received_signal) throw Interrupted();
            if (!model) throw std::runtime_error("model loading failed; see engine diagnostics above");
            vocab = llama_model_get_vocab(model.get());
            if (!vocab || llama_vocab_n_tokens(vocab) <= 0) throw std::runtime_error("model has no usable vocabulary");
            if (llama_model_has_encoder(model.get())) throw std::invalid_argument("encoder models are outside this client's text-only scope");
            std::cerr << "[longbook27] model loaded; context=" << options.context
                      << " batch=" << options.batch << " ubatch=" << options.ubatch
                      << " gpu_layers=" << options.gpu_layers << '\n';
        } catch (...) {
            context.reset();
            model.reset();
            llama_backend_free();
            llama_log_set(nullptr, nullptr);
            throw;
        }
    }

    ~Impl() {
        context.reset();
        model.reset();
        llama_backend_free();
        llama_log_set(nullptr, nullptr);
    }

    void fresh_context() {
        // A fresh context avoids partial recurrent-state edits and dirty abort recovery.
        context.reset();
        auto p = llama_context_default_params();
        p.n_ctx = static_cast<std::uint32_t>(options.context);
        p.n_batch = static_cast<std::uint32_t>(options.batch);
        p.n_ubatch = static_cast<std::uint32_t>(options.ubatch);
        p.n_seq_max = 1;
        p.n_rs_seq = 0;
        p.n_threads = options.threads;
        p.n_threads_batch = options.threads;
        p.abort_callback = abort_inference;
        p.no_perf = true;
        context.reset(llama_init_from_model(model.get(), p));
        if (received_signal) throw Interrupted();
        if (!context) throw std::runtime_error("context allocation failed; reduce context/batch/GPU layers or free memory");
        if (llama_n_ctx(context.get()) < static_cast<std::uint32_t>(options.context)) throw std::runtime_error("engine allocated less context than requested");
    }

    void decode(llama_token * tokens, int count, int first_position, bool last_logits) {
        if (received_signal) throw Interrupted();
        // Explicit positions and sequence IDs make prefill chunk boundaries unambiguous.
        std::vector<llama_pos> positions(static_cast<std::size_t>(count));
        std::vector<std::int32_t> n_seq(static_cast<std::size_t>(count), 1);
        std::vector<llama_seq_id> seq(static_cast<std::size_t>(count), 0);
        std::vector<llama_seq_id *> seq_ptr(static_cast<std::size_t>(count));
        std::vector<std::int8_t> logits(static_cast<std::size_t>(count), 0);
        for (int i = 0; i < count; ++i) { positions[i] = first_position + i; seq_ptr[i] = &seq[i]; }
        if (last_logits) logits.back() = 1;
        llama_batch batch{};
        batch.n_tokens = count;
        batch.token = tokens;
        batch.pos = positions.data();
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ptr.data();
        batch.logits = logits.data();
        const int code = llama_decode(context.get(), batch);
        if (received_signal || code == 2) throw Interrupted();
        if (code != 0) {
            context.reset();
            throw std::runtime_error("decode failed with code=" + std::to_string(code) + "; recurrent state discarded");
        }
    }
};

Client::Client(const Options & options) : impl_(std::make_unique<Impl>(options)) {}
Client::~Client() = default;

Result Client::generate(const std::vector<Message> & messages, std::ostream & output) {
    auto & engine = *impl_;
    const auto & o = engine.options;
    if (messages.empty()) throw std::invalid_argument("prompt must not be empty");
    const auto prompt = o.raw ? messages.back().content : render_chat(messages, o.thinking);
    auto tokens = tokenize(engine.vocab, prompt, !o.raw);
    if (tokens.size() + static_cast<std::size_t>(o.predict) > static_cast<std::size_t>(o.context)) {
        throw std::invalid_argument("context budget exceeded: prompt_tokens=" + std::to_string(tokens.size())
            + " + predict=" + std::to_string(o.predict) + " > ctx_size=" + std::to_string(o.context) + "; input was not truncated");
    }
    engine.fresh_context();
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    Sampler sampler(llama_sampler_chain_init(sampler_params), llama_sampler_free);
    if (!sampler) throw std::runtime_error("sampler allocation failed");
    if (o.temperature == 0) llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(o.top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(o.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(o.top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(o.seed));
    }
    Result result;
    result.prompt_tokens = static_cast<int>(tokens.size());
    const auto prefill_start = Clock::now();
    const int batch_size = std::min(o.batch, static_cast<int>(llama_n_batch(engine.context.get())));
    for (int pos = 0; pos < result.prompt_tokens;) {
        const int count = std::min(batch_size, result.prompt_tokens - pos);
        engine.decode(tokens.data() + pos, count, pos, pos + count == result.prompt_tokens);
        pos += count;
    }
    const auto generate_start = Clock::now();
    result.prefill_ms = milliseconds(prefill_start, generate_start);
    for (int i = 0; i < o.predict; ++i) {
        if (received_signal) { result.stop = "interrupt"; break; }
        const auto * logits = llama_get_logits_ith(engine.context.get(), -1);
        if (!logits) throw std::runtime_error("engine returned no final-token logits");
        const auto vocab_size = llama_vocab_n_tokens(engine.vocab);
        for (int t = 0; t < vocab_size; ++t) if (!std::isfinite(logits[t])) throw std::runtime_error("non-finite logits; generation stopped");
        auto token = llama_sampler_sample(sampler.get(), engine.context.get(), -1);
        if (llama_vocab_is_eog(engine.vocab, token)) { result.stop = "eog"; break; }
        const auto piece = token_piece(engine.vocab, token);
        output.write(piece.data(), static_cast<std::streamsize>(piece.size()));
        output.flush();
        if (!output) throw std::runtime_error("failed to write generated output");
        result.text += piece;
        ++result.generated_tokens;
        if (i + 1 < o.predict) engine.decode(&token, 1, result.prompt_tokens + i, true);
    }
    result.generation_ms = milliseconds(generate_start, Clock::now());
    // Do not retain a partially processed context after generation or cancellation.
    engine.context.reset();
    return result;
}
} // namespace longbook
