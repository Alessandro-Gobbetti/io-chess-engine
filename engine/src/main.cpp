/**
 * @file main.cpp
 * @brief Chess Engine Entry Point.
 *
 * Initializes the chess engine and handles command line arguments to
 * start the UCI protocol interface.
 */

#include "uci/UCI.h"
#include <cstring>
#include <iostream>
#include <string>

void printUsage(const char *programName) {
  std::cerr << "Usage: " << programName << " [options]" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  --model <path>    Path to native model bundle directory "
               "(must contain native_weights.bin; required for Neural eval)"
            << std::endl;
  std::cerr << "  --simple-eval     Use simple evaluation (no model needed)"
            << std::endl;
  std::cerr << "  --syzygy <path>   Path to Syzygy tablebases (optional; can "
               "also use SYZYGY_PATH env)"
            << std::endl;
  std::cerr << "  --book <path>     Path to Polyglot opening book (optional)"
            << std::endl;
  std::cerr << "  --book2 <path>    Path to secondary Polyglot opening book "
               "(optional)"
            << std::endl;
  std::cerr << "  --help           Show this help message" << std::endl;
}

int main(int argc, char *argv[]) {
  unsigned int hwCores = std::thread::hardware_concurrency();
  std::cout << "info string Detected Hardware Cores: " << hwCores << std::endl;

  std::string modelPath;
  bool useSimpleEval = false;
  std::string tbPath;
  std::string bookPath;
  std::string bookPath2;
  const char *envTb = std::getenv("SYZYGY_PATH");
  if (envTb)  
    tbPath = envTb;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (std::strcmp(argv[i], "--simple-eval") == 0) {
      useSimpleEval = true;
    } else if (std::strcmp(argv[i], "--syzygy") == 0 && i + 1 < argc) {
      tbPath = argv[++i];
    } else if (std::strcmp(argv[i], "--book") == 0 && i + 1 < argc) {
      bookPath = argv[++i];
    } else if (std::strcmp(argv[i], "--book2") == 0 && i + 1 < argc) {
      bookPath2 = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      printUsage(argv[0]);
      return 0;
    }
  }

  // Check required arguments
  if (modelPath.empty() && !useSimpleEval) {
    std::cerr << "Error: Either --model or --simple-eval must be specified."
              << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  if (!modelPath.empty() && useSimpleEval) {
    std::cerr
        << "Error: Cannot use both --model and --simple-eval at the same time."
        << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Create UCI protocol handler and run main loop
  try {
    if (useSimpleEval) {
      std::cout << "info string Using simple evaluation mode" << std::endl;
    }
    UciProtocol uci(modelPath, useSimpleEval, tbPath, bookPath, bookPath2);
    uci.loop();
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}