#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "SubmodularFunction.h"
#include "kernels/RBFKernel.h"
#include "functions/IVM.h"
#include "functions/FastIVM.h"
#include "Greedy.h"

namespace py = pybind11;

class PySubmodularFunction : public SubmodularFunction {
public:
    SubmodularFunction* clone() {
        PYBIND11_OVERRIDE_PURE(
            SubmodularFunction*,        /* Return type */
            SubmodularFunction,         /* Parent class */
            clone,                      /* Name of function in C++ (must match Python name) */
        );
    }

    data_t operator()(std::vector<std::vector<data_t>> const &solution) const {
        PYBIND11_OVERRIDE_PURE_NAME(
            data_t,                     /* Return type */
            SubmodularFunction,         /* Parent class */
            "__call__",                 /* Name of method in Python */
            operator(),                 /* Name of function in C++ */
            solution                    /* Argument(s) */
        );
    }

    data_t peek(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {
       PYBIND11_OVERRIDE(
           data_t,
           SubmodularFunction,
           peek,
           cur_solution, 
           x
       );
    }

    void update(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {
        PYBIND11_OVERRIDE(
           void,
           SubmodularFunction,
           update,
           cur_solution, 
           x
       );
    }
};

PYBIND11_MODULE(PySSM, m) {
    py::class_<RBFKernel>(m, "RBFKernel")
        .def(py::init<data_t, data_t>(), py::arg("sigma") = 1.0, py::arg("scale") = 1.0)
        .def(py::init<float>(), py::arg("sigma") = 1.0)
        .def(py::init<>())
        .def("__call__", &RBFKernel::operator());

    py::class_<SubmodularFunction, PySubmodularFunction, std::shared_ptr<SubmodularFunction>>(m, "SubmodularFunction")
        .def(py::init<>())
        .def("peek", &SubmodularFunction::peek, py::arg("cur_solution"), py::arg("x"))
        .def("update", &SubmodularFunction::update, py::arg("cur_solution"), py::arg("x"))
        .def("__call__", &SubmodularFunction::operator());
        // .def("clone", &SubmodularFunction::clone);

    py::class_<IVM, SubmodularFunction, std::shared_ptr<IVM> >(m, "IVM")
        .def(py::init<std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(), py::arg("kernel"), py::arg("sigma"))
        .def("peek", &IVM::peek, py::arg("cur_solution"), py::arg("x"))
        .def("update", &IVM::update, py::arg("cur_solution"), py::arg("x"))
        .def("__call__", &IVM::operator());

    // py::class_<Greedy>(m, "Greedy") 
    //     .def(py::init<unsigned int, std::shared_ptr<SubmodularFunction &>>, py::arg("K"), py::arg("f"))
    //     .def("get_solution", &Greedy::get_solution)
    //     .def("get_fval", &Greedy::get_fval)
    //     .def("fit", &Greedy::fit, py::arg("X"));

    py::class_<Greedy>(m, "Greedy") 
        //.def(py::init<unsigned int, std::shared_ptr<SubmodularFunction>>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, SubmodularFunction&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)> >(), py::arg("K"), py::arg("f"))
        .def("get_solution", &Greedy::get_solution)
        .def("get_fval", &Greedy::get_fval)
        .def("fit", &Greedy::fit, py::arg("X"));
}


// PySSMM
// PYBIND11_MODULE(PySSMM, m) {
//     py::class_<RBFKernel>(m, "RBFKernel")
//         .def("update", &IVM::update);
// }
