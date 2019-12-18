#include "generator/include/deterministic.h"

#include <map>
#include <array>

void generate_cyclic(Signal& signal, 
                     const uint32_t mod,
                     const uint32_t offset) {
    for (int i = 0; i < signal.length(); i++) {
        if ((i % mod == 0) && (i + offset < signal.length())) {
            signal.data()[i + offset] = true;
        } 
    }
}

// Dash = 1, Dot = 0
// Construct the encoding table

typedef std::map<char, std::vector<bool>> morse_code_table_t;
typedef std::pair<char, std::vector<bool>> morse_code_table_entry_t;

// Z = Zero, O = One
const bool Z = false;
const bool O = true;

morse_code_table_t morse_code_table = {
    // Numbers
    std::make_pair('1', std::vector<bool>({Z, O, O, O, O})),
    std::make_pair('2', std::vector<bool>({Z, Z, O, O, O})),
    std::make_pair('3', std::vector<bool>({Z, Z, Z, O, O})),
    std::make_pair('4', std::vector<bool>({Z, Z, Z, Z, O})),
    std::make_pair('5', std::vector<bool>({Z, Z, Z, Z, Z})),
    std::make_pair('6', std::vector<bool>({O, Z, Z, Z, Z})),
    std::make_pair('7', std::vector<bool>({O, O, Z, Z, Z})),
    std::make_pair('8', std::vector<bool>({O, O, O, Z, Z})),
    std::make_pair('9', std::vector<bool>({O, O, O, O, Z})),
    std::make_pair('0', std::vector<bool>({O, O, O, O, O})),

    // Letters
    std::make_pair('a', std::vector<bool>({Z, O})),
    std::make_pair('b', std::vector<bool>({O, Z, Z, Z})),
    std::make_pair('c', std::vector<bool>({O, Z, O, Z})),
    std::make_pair('d', std::vector<bool>({O, Z, Z})),
    std::make_pair('e', std::vector<bool>({Z})),
    std::make_pair('f', std::vector<bool>({Z, Z, O, Z})),
    std::make_pair('g', std::vector<bool>({O, O, Z})),
    std::make_pair('h', std::vector<bool>({Z, Z, Z, Z})),
    std::make_pair('i', std::vector<bool>({Z, Z})),
    std::make_pair('j', std::vector<bool>({Z, O, O, O})),
    std::make_pair('k', std::vector<bool>({O, Z, O})),
    std::make_pair('l', std::vector<bool>({Z, O, Z, Z})),
    std::make_pair('m', std::vector<bool>({O, O})),
    std::make_pair('n', std::vector<bool>({O, Z})),
    std::make_pair('o', std::vector<bool>({O, O, O})),
    std::make_pair('p', std::vector<bool>({Z, O, O, Z})),
    std::make_pair('q', std::vector<bool>({O, O, Z, O})),
    std::make_pair('r', std::vector<bool>({Z, O, Z})),
    std::make_pair('s', std::vector<bool>({Z, Z, Z})),
    std::make_pair('t', std::vector<bool>({O})),
    std::make_pair('u', std::vector<bool>({Z, Z, O})),
    std::make_pair('v', std::vector<bool>({Z, Z, Z, O})),
    std::make_pair('w', std::vector<bool>({Z, O, O})),
    std::make_pair('x', std::vector<bool>({O, Z, Z, O})),
    std::make_pair('y', std::vector<bool>({O, Z, O, O})),
    std::make_pair('z', std::vector<bool>({O, O, Z, Z})),
};

void generate_morse_encode(Signal& signal,
                           const std::string& str,
                           const std::vector<bool>& separator_pattern) {
    // Construct a zero length signal when we have no characters to encode
    if (str.size() == 0) {
        signal.data().resize(0);
        return;
    }
    
    ///////////////////////////////////////////
    // Allocate enough bits for the worst case scenario (each character requiring 6 bits)
    signal.data().resize((6 + separator_pattern.size()) * str.size());
    signal.zero();

    std::uint64_t time_index = 0;

    for (char c: str) {
        c = std::tolower(c);
        // Treat unknown characters as zeros
        if (morse_code_table.count(c) == 0)
            c = '0';
        
        // Find the code
        auto code = morse_code_table.find(c)->second;

        // Merge the morse code and separator pattern
        std::vector<bool> encoded = code;
        encoded.insert(encoded.end(), 
                       separator_pattern.begin(),
                       separator_pattern.end());

        // Put them into the signal
        std::copy(encoded.begin(), encoded.end(), 
                  signal.data().begin() + time_index);

        time_index += encoded.size();
    }

    // Remove needless bits at the end. This way we save memory.
    signal.data().resize(time_index);
}