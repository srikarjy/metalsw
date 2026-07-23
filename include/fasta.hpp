#pragma once

#include <string>
#include <vector>

namespace metalsw {

struct FastaRecord {
    std::string id;
    std::string sequence;
};

// Throws std::runtime_error if the file cannot be opened or contains no records.
std::vector<FastaRecord> parseFasta(const std::string &path);

}  // namespace metalsw
