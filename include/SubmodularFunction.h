#ifndef SUBMODULARFUNCTION_H
#define SUBMODULARFUNCTION_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>

#include "DataTypeHandling.h"

class SubmodularFunction {
public:
    // virtual SubmodularFunction* clone() = 0;

    virtual data_t operator()(std::vector<std::vector<data_t>> const &solution) const = 0;

    virtual data_t peek(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {
        cur_solution.push_back(x);
        data_t ftmp = this->operator()(cur_solution);
        cur_solution.pop_back();
        return ftmp;
    }

    virtual void update(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {}

    virtual std::shared_ptr<SubmodularFunction> clone() const = 0;

    virtual ~SubmodularFunction() {}
};

#endif // SUBMODULARFUNCTION_H
