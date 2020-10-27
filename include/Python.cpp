#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "SubmodularFunction.h"
#include "kernels/RBFKernel.h"
#include "functions/IVM.h"
#include "functions/FastIVM.h"
#include "Greedy.h"
#include "Random.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"
#include "ThreeSieves.h"

namespace py = pybind11;

class PySubmodularFunction : public SubmodularFunction {
public:
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

    ~PySubmodularFunction() {
        std::cout << "D'TOR CALLED" << std::endl;
    }

    // See https://github.com/pybind/pybind11/issues/1049
    std::shared_ptr<SubmodularFunction> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        std::vector<std::vector<data_t>> X({{1.0,2.0}});
        std::vector<data_t> x({1.0,1.0});

        auto keep_python_state_alive = std::make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PySubmodularFunction*>();
        std::cout << "RETURN COPY" << std::endl;

        std::shared_ptr<SubmodularFunction> newobj = std::shared_ptr<SubmodularFunction>(keep_python_state_alive, ptr);
        std::cout << "adress: " << newobj << std::endl;

        // aliasing shared_ptr: points to `A_trampoline* ptr` but refcounts the Python object
        return newobj;
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
        .def("__call__", &SubmodularFunction::operator())
        .def("clone", &SubmodularFunction::clone, py::return_value_policy::reference);

    py::class_<IVM, SubmodularFunction, std::shared_ptr<IVM> >(m, "IVM")
        .def(py::init<std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(), py::arg("kernel"), py::arg("sigma"))
        .def("peek", &IVM::peek, py::arg("cur_solution"), py::arg("x"))
        .def("update", &IVM::update, py::arg("cur_solution"), py::arg("x"))
        .def("__call__", &IVM::operator())
        .def("clone", &IVM::clone, py::return_value_policy::reference);

    py::class_<FastIVM, IVM, SubmodularFunction, std::shared_ptr<FastIVM> >(m, "FastIVM")
        .def(py::init<unsigned int, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(),  py::arg("K"),  py::arg("kernel"), py::arg("sigma"))
        .def("peek", &FastIVM::peek, py::arg("cur_solution"), py::arg("x"))
        .def("update", &FastIVM::update, py::arg("cur_solution"), py::arg("x"))
        .def("__call__", &FastIVM::operator())
        .def("clone", &FastIVM::clone, py::return_value_policy::reference);

    py::class_<Greedy>(m, "Greedy") 
        //.def(py::init<unsigned int, std::shared_ptr<SubmodularFunction>>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, SubmodularFunction&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)> >(), py::arg("K"), py::arg("f"))
        .def("get_solution", &Greedy::get_solution)
        .def("get_fval", &Greedy::get_fval)
        .def("fit", &Greedy::fit, py::arg("X"));
    
    py::class_<Random>(m, "Random") 
        .def(py::init<unsigned int, SubmodularFunction&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)> >(), py::arg("K"), py::arg("f"))
        .def("get_solution", &Random::get_solution)
        .def("get_fval", &Random::get_fval)
        .def("fit", &Random::fit, py::arg("X"))
        .def("next", &Random::next, py::arg("x"));

    py::class_<SieveStreaming>(m, "SieveStreaming") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreaming::get_solution)
        .def("get_fval", &SieveStreaming::get_fval)
        .def("fit", &SieveStreaming::fit, py::arg("X"))
        .def("next", &SieveStreaming::next, py::arg("x"));
    
    py::class_<SieveStreamingPP>(m, "SieveStreamingPP") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreamingPP::get_solution)
        .def("get_fval", &SieveStreamingPP::get_fval)
        .def("fit", &SieveStreamingPP::fit, py::arg("X"))
        .def("next", &SieveStreamingPP::next, py::arg("x"));
    
    // py::enum_<ThreeSieves::THRESHOLD_STRATEGY>(m, "ThresholdStrategy")
    //     .value("SIEVE", ThreeSieves::THRESHOLD_STRATEGY::SIEVE)
    //     .value("CONSTANT", ThreeSieves::THRESHOLD_STRATEGY::CONSTANT)
    //     .export_values();
        
    py::class_<ThreeSieves>(m, "ThreeSieves") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def("get_solution", &ThreeSieves::get_solution)
        .def("get_fval", &ThreeSieves::get_fval)
        .def("fit", &ThreeSieves::fit, py::arg("X"))
        .def("next", &ThreeSieves::next, py::arg("x"));
}


// PySSMM
// PYBIND11_MODULE(PySSMM, m) {
//     py::class_<RBFKernel>(m, "RBFKernel")
//         .def("update", &IVM::update);
// }
