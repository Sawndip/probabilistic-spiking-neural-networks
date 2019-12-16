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

typedef std::map<char, std::vector<bool>> morse_code_table_t;

// Dash = 1, Dot = 0
void __fill_morse_code_table(morse_code_table_t& morse_code_table) {
    const bool Z = false;
    const bool O = true;

    // Numbers
    morse_code_table['1'] = {Z, O, O, O, O};
    morse_code_table['2'] = {Z, Z, O, O, O};
    morse_code_table['3'] = {Z, Z, Z, O, O};
    morse_code_table['4'] = {Z, Z, Z, Z, O};
    morse_code_table['5'] = {Z, Z, Z, Z, Z};
    morse_code_table['6'] = {O, Z, Z, Z, Z};
    morse_code_table['7'] = {O, O, Z, Z, Z};
    morse_code_table['8'] = {O, O, O, Z, Z};
    morse_code_table['9'] = {O, O, O, O, Z};
    morse_code_table['0'] = {O, O, O, O, O};

    // Letters
    morse_code_table['a'] = {Z, O};
    morse_code_table['b'] = {O, Z, Z, Z};
    morse_code_table['c'] = {O, Z, O, Z};
    morse_code_table['d'] = {O, Z, Z};
    morse_code_table['e'] = {Z};
    morse_code_table['f'] = {Z, Z, O, Z};
    morse_code_table['g'] = {O, O, Z};
    morse_code_table['h'] = {Z, Z, Z, Z};
    morse_code_table['i'] = {Z, Z};
    morse_code_table['j'] = {Z, O, O, O};
    morse_code_table['k'] = {O, Z, O};
    morse_code_table['l'] = {Z, O, Z, Z};
    morse_code_table['m'] = {O, O};
    morse_code_table['n'] = {O, Z};
    morse_code_table['o'] = {O, O, O};
    morse_code_table['p'] = {Z, O, O, Z};
    morse_code_table['q'] = {O, O, Z, O};
    morse_code_table['r'] = {Z, O, Z};
    morse_code_table['s'] = {Z, Z, Z};
    morse_code_table['t'] = {O};
    morse_code_table['u'] = {Z, Z, O};
    morse_code_table['v'] = {Z, Z, Z, O};
    morse_code_table['w'] = {Z, O, O};
    morse_code_table['x'] = {O, Z, Z, O};
    morse_code_table['y'] = {O, Z, O, O};
    morse_code_table['z'] = {O, O, Z, Z};
}

void generate_morse_encode(Signal& signal,
                           const std::string& str,
                           const std::vector<bool>& separator_pattern) {
    // Construct a zero length signal when we have no characters to encode
    if (str.size() == 0) {
        signal.data().resize(0);
        return;
    }

    // Construct the encoding table
    morse_code_table_t morse_code_table;
    __fill_morse_code_table(morse_code_table);

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