#include "fasta.hpp"

#include <fstream>
#include <stdexcept>

namespace metalsw {

std::vector<FastaRecord> parseFasta(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open FASTA file: " + path);
    }

    std::vector<FastaRecord> records;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '>') {
            records.push_back(FastaRecord{line.substr(1), ""});
        } else if (!records.empty()) {
            records.back().sequence += line;
        }
    }

    if (records.empty()) {
        throw std::runtime_error("no records found in FASTA file: " + path);
    }
    return records;
}

}  // namespace metalsw
