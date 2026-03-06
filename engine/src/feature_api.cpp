#include "FeatureExtractor.hpp"
#include "chess.hpp"

#include <cstdint>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

static ChessInput g_features;

#include "ExpertRouter.hpp"

extern "C" {
#if defined(__EMSCRIPTEN__) && defined(FEATURE_WASM_MAIN)
int main() {
  return 0;
}
#endif

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int extract_features(const char *fen) {
  if (!fen) {
    return 0;
  }

  chess::Board board;
  if (!board.setFen(fen)) {
    return 0;
  }

  FeatureExtractor::fill_input(board, g_features);
  return 1;
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
std::uintptr_t get_feature_layers_ptr() {
  return reinterpret_cast<std::uintptr_t>(&g_features.layers[0][0]);
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
std::uintptr_t get_feature_globals_ptr() {
  return reinterpret_cast<std::uintptr_t>(&g_features.global[0]);
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int get_feature_layers_len() {
  return 32 * 64;
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int get_feature_globals_len() {
  return 16;
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void get_expert_weights(float* output) {
  ExpertRouter::ExpertWeights weights;
  // Use 0.0f for eval_cp as the routing visual doesn't need the CP-dependent aux gates (Survivor/Killer) 
  // to show the base experts, OR we can accept it as a param if we want complete accuracy.
  // For the visual "Routing" card, we care most about the 4 base experts.
  // Let's pass 0.0f for now to keep API simple.
  ExpertRouter::compute_weights(g_features, 0.0f, weights);
  
  // Copy 6 weights to output buffer (caller must allocate 6 floats)
  for(int i=0; i<6; ++i) {
    output[i] = weights.weights[i];
  }
}

}
