# Primary project goals

## Implementation of ideas from "An Introduction to Probabilistic Spiking Neural Networks"

This is a paper by Hyeryung Jang, Osvaldo Simeone, Brian Gardner, and Andre Gruning available at https://arxiv.org/abs/1910.01059v1

For convenience the paper is also available in this repository.

### 1. Development of the fully observed case on dummy data

### 2. Development of the partially observed case on dummy data

### 3. (Possibly) Replication of results from the paper

## Development of string embeddings

As potential inspiration come the two popular papers:
- word2vec - https://arxiv.org/abs/1301.3781
- ELMo - https://arxiv.org/abs/1802.05365

The goal is that given a token out of some finite dictionary a binary signals of spikes is generated for that token. Then by simple concatenation of token signals a big binary time signal is constructed for a document of tokens. Between the encoded tokens a inter-symbol sequence of all 1s or all 0s should be added.

The encoding and decoding of tokens should happen via the learned embeddings.

## Development of (rudimentary) propositional logic capabilities

The specific goal is the development of modus ponens capability:
- https://en.wikipedia.org/wiki/Modus_ponens

### Example

p: Aristotle can swim
q: Fish can swim
----------------
r: Aristotle is a fish

The spiking NN for this task will use token embeddings of **p** and **q** for the two input neurons and will output a single binary signal. Using token embeddings the output binary signal will be decoded to the string **r**

# Secondary project goals

## Reusability of the written C++ code as a library for other projects.

## Wrappers for python, MATLAB, R, etc