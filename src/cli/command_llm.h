#pragma once
#include "cli/cli_context.h"
#include "cli/formatter.h"
#include <string>

int RunLlmList(Formatter &fmt, const CliContext &ctx);
int RunLlmAdd(const std::string &name, const std::string &base_url,
              const std::string &model, const std::string &api_key,
              int timeout_ms, Formatter &fmt, const CliContext &ctx);
int RunLlmRemove(const std::string &name, bool force, Formatter &fmt,
                 const CliContext &ctx);
