#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <chrono>

#include "functions/FastIVM.h"
#include "functions/kernels/RBFKernel.h"
#include "Greedy.h"
#include "Random.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"
#include "ThreeSieves.h"
#include "DataTypeHandling.h"

std::vector<std::vector<data_t>> read_arff(std::string const& path) {
    std::vector<std::vector<data_t>> X; 

    std::string line;
    std::ifstream file(path);

    if (file.is_open()) {
        while (std::getline(file, line)) {
            // Skip every meta information
            if (line.size() > 0 && line[0] != '@' && line != "\r") {
                std::vector<data_t> x;
                std::stringstream ss(line);
                std::string entry;
                // All entries are float, but the last one (the label, string) and the second to last(the id, integer). Skip both.
                while (std::getline(ss, entry, ',') && x.size() < 78) {
                    if (entry.size() > 0) { //&& entry[0] != '\''
                        x.push_back( static_cast<float>(atof(entry.c_str())) );
                    }
                }
                if (X.size() > 0 && x.size() != X[0].size()) {
                    std::cout << "Size mismatch detected. Ignoring line." << std::endl;
                } else {
                    X.push_back(x);
                }
            }
        }
        file.close();
    }

    return X;
}

auto evaluate_optimizer(SubmodularOptimizer &opt, std::vector<std::vector<data_t>> &X) {
    auto start = std::chrono::steady_clock::now();
    opt.fit(X);
    auto end = std::chrono::steady_clock::now();   
    std::chrono::duration<double> runtime_seconds = end-start;
    auto fval = opt.get_fval();

    return std::make_tuple(fval, runtime_seconds.count());
}

std::string to_string(std::vector<std::vector<data_t>> const &solution) {
    std::string s;

    for (auto x : solution) {
        for (auto xi : x) {
            s += std::to_string(xi) + " ";
        }
        s += "\n";
    }

    return s;
}

int main() {
    std::cout << "Reading data" << std::endl;
    auto data = read_arff("../experiments/kddcup99/data/KDDCup99/KDDCup99_withoutdupl_norm_1ofn.arff");

    // auto cnt = 0;
    // for (auto x : data) {
    //     for (auto xi : x) {
    //         std::cout << xi << " ";
    //     }
    //     std::cout << std::endl;
    //     ++cnt;
    //     if (cnt > 10) break;
    // }

    auto K = 50;
    FastIVM fastIVM(K, RBFKernel( std::sqrt(data[0].size()), 1.0) , 1.0);

    std::cout << "Selecting " << K << " representatives via Greedy" << std::endl;
    Greedy greedy(K, fastIVM);
    auto res = evaluate_optimizer(greedy, data);
    std::cout << "\t fval:\t\t" << std::get<0>(res) << "\n\t runtime:\t" << std::get<1>(res) << "s\n\n" << std::endl;

    // std::cout << "Selecting " << K << " representatives via Random with seed = 0" << std::endl;
    // Random random0(K, fastIVM, 0);
    // res = evaluate_optimizer(random0, data);
    // std::cout << "\t fval:\t\t" << std::get<0>(res) << "\n\t runtime:\t" << std::get<1>(res) << "s\n\n" << std::endl;

    // std::cout << "Selecting " << K << " representatives via SieveStreaming" << std::endl;
    // SieveStreaming sieve(K, fastIVM, 1.0, 0.01);
    // res = evaluate_optimizer(sieve, data);
    // std::cout << "\t fval:\t\t" << std::get<0>(res) << "\n\t runtime:\t" << std::get<1>(res) << "s\n\n" << std::endl;

    // std::cout << "Selecting " << K << " representatives via SieveStreaming++" << std::endl;
    // SieveStreamingPP sievepp(K, fastIVM, 1.0, 0.01);
    // res = evaluate_optimizer(sievepp, data);
    // std::cout << "\t fval:\t\t" << std::get<0>(res) << "\n\t runtime:\t" << std::get<1>(res) << "\n\n" << std::endl;

    // std::cout << "Selecting " << K << " representatives via ThreeSieves" << std::endl;
    // ThreeSieves three(K, fastIVM, 1.0, 0.01, ThreeSieves::THRESHOLD_STRATEGY::SIEVE, 1000);
    // res = evaluate_optimizer(three, data);
    // std::cout << "\t fval:\t\t" << std::get<0>(res) << "\n\t runtime:\t" << std::get<1>(res) << "s\n\n" << std::endl;
}