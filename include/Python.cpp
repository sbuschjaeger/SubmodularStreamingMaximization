#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "SubmodularFunction.h"
#include "functions/kernels/RBFKernel.h"
#include "functions/kernels/Kernel.h"
#include "functions/IVM.h"
#include "functions/FastIVM.h"
#include "Greedy.h"
#include "Random.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"
#include "ThreeSieves.h"
#include "Salsa.h"
#include "IndependentSetImprovement.h"

namespace py = pybind11;

class PySubmodularFunction : public SubmodularFunction {
public:
    data_t operator()(std::vector<std::vector<data_t>> const &solution) const override {
        PYBIND11_OVERRIDE_PURE_NAME(
            data_t,                     /* Return type */
            SubmodularFunction,         /* Parent class */
            "__call__",                 /* Name of method in Python */
            operator(),                 /* Name of function in C++ */
            solution                    /* Argument(s) */
        );
    }

    data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
       PYBIND11_OVERRIDE_PURE(
           data_t,
           SubmodularFunction,
           peek,
           cur_solution, 
           x,
           pos
       );
    }

    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        PYBIND11_OVERRIDE_PURE(
           void,
           SubmodularFunction,
           update,
           cur_solution, 
           x,
           pos
       );
    }

    ~PySubmodularFunction() {
    }

    // See https://github.com/pybind/pybind11/issues/1049
    std::shared_ptr<SubmodularFunction> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = std::make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PySubmodularFunction*>();

        std::shared_ptr<SubmodularFunction> newobj = std::shared_ptr<SubmodularFunction>(keep_python_state_alive, ptr);

        // aliasing shared_ptr: points to `A_trampoline* ptr` but refcounts the Python object
        return newobj;
    }
};

class PyKernel : public Kernel {
public:
    data_t operator()(std::vector<data_t> const &x1, std::vector<data_t> const &x2) const override {
        PYBIND11_OVERRIDE_PURE_NAME(
            data_t,                     /* Return type */
            Kernel,                     /* Parent class */
            "__call__",                 /* Name of method in Python */
            operator(),                 /* Name of function in C++ */
            x1,                         /* Argument(s) */
            x2
        );
    }

    ~PyKernel() {}

    // See https://github.com/pybind/pybind11/issues/1049
    std::shared_ptr<Kernel> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = std::make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyKernel*>();

        std::shared_ptr<Kernel> newobj = std::shared_ptr<Kernel>(keep_python_state_alive, ptr);

        // aliasing shared_ptr: points to `A_trampoline* ptr` but refcounts the Python object
        return newobj;
    }
};

// data_t fit_greedy_on_ivm(unsigned int K, data_t sigma, data_t scale, data_t epsilon, std::vector<std::vector<data_t>> const &X) {
//     FastIVM fastIVM(K, RBFKernel(sigma, scale), epsilon);
//     Greedy greedy(K, fastIVM);
//     greedy.fit(X);
//     return greedy.get_fval();
// }

// data_t fit_greedy_on_ivm_2(unsigned int K, data_t sigma, data_t scale, data_t epsilon) {
//     return 1.0;
//     // FastIVM fastIVM(K, RBFKernel(sigma, scale), epsilon);
//     // Greedy greedy(K, fastIVM);
//     // greedy.fit(X);
//     // return greedy.get_fval();
// }

// PYBIND11_MAKE_OPAQUE(std::vector<data_t>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<data_t>>);

PYBIND11_MODULE(PySSM, m) {
    // m.def("fit_greedy_on_ivm", &fit_greedy_on_ivm, 
    //     py::arg("K"), 
    //     py::arg("sigma"),
    //     py::arg("scale"),
    //     py::arg("epsilon"),
    //     py::arg("X")
    // );

    // m.def("fit_greedy_on_ivm_2", &fit_greedy_on_ivm_2, 
    //     py::arg("K"), 
    //     py::arg("sigma"),
    //     py::arg("scale"),
    //     py::arg("epsilon")
    // );

    // py::bind_vector<std::vector<data_t>>(m, "Vector");
    // py::bind_vector<std::vector<std::vector<data_t>>>(m, "Matrix");

    py::class_<Kernel, PyKernel, std::shared_ptr<Kernel>>(m, "Kernel")
        .def(py::init<>())
        .def("__call__", &Kernel::operator())
        .def("clone", &Kernel::clone, py::return_value_policy::reference);

    py::class_<RBFKernel, Kernel, std::shared_ptr<RBFKernel>>(m, "RBFKernel")
        .def(py::init<data_t, data_t>(), py::arg("sigma") = 1.0, py::arg("scale") = 1.0)
        .def(py::init<data_t>(), py::arg("sigma") = 1.0)
        .def(py::init<>())
        .def("__call__", &RBFKernel::operator())
        .def("clone", &RBFKernel::clone, py::return_value_policy::reference);

    py::class_<SubmodularFunction, PySubmodularFunction, std::shared_ptr<SubmodularFunction>>(m, "SubmodularFunction")
        .def(py::init<>())
        .def("peek", &SubmodularFunction::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &SubmodularFunction::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &SubmodularFunction::operator())
        .def("clone", &SubmodularFunction::clone, py::return_value_policy::reference);

    py::class_<IVM, SubmodularFunction, std::shared_ptr<IVM> >(m, "IVM")
        //.def(py::init<std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(), py::arg("kernel"), py::arg("sigma"))
        .def(py::init<Kernel const &, data_t>(), py::arg("kernel"), py::arg("sigma") = 1.0)
        .def("peek", &IVM::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &IVM::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &IVM::operator())
        .def("clone", &IVM::clone, py::return_value_policy::reference);

    py::class_<FastIVM, IVM, SubmodularFunction, std::shared_ptr<FastIVM> >(m, "FastIVM")
        //.def(py::init<unsigned int, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(),  py::arg("K"),  py::arg("kernel"), py::arg("sigma"))
        .def(py::init<unsigned int, Kernel const &, data_t>(),  py::arg("K"),  py::arg("kernel"), py::arg("sigma") = 1.0)
        .def("peek", &FastIVM::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &FastIVM::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &FastIVM::operator())
        .def("clone", &FastIVM::clone, py::return_value_policy::reference);

    py::class_<Greedy>(m, "Greedy") 
        //.def(py::init<unsigned int, std::shared_ptr<SubmodularFunction>>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, SubmodularFunction&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)> >(), py::arg("K"), py::arg("f"))
        .def("get_solution", &Greedy::get_solution)
        .def("get_fval", &Greedy::get_fval)
        .def("get_num_candidate_solutions", &Greedy::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Greedy::get_num_elements_stored)
        .def("fit", &Greedy::fit, py::arg("X"), py::arg("iterations") = 1);
    
    py::class_<Random>(m, "Random") 
        .def(py::init<unsigned int, SubmodularFunction&, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("seed")= 0)
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("seed") = 0)
        .def("get_solution", &Random::get_solution)
        .def("get_fval", &Random::get_fval)
        .def("get_num_candidate_solutions", &Random::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Random::get_num_elements_stored)
        .def("fit", &Random::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &Random::next, py::arg("x"));

    py::class_<IndependentSetImprovement>(m, "IndependentSetImprovement") 
        .def(py::init<unsigned int, SubmodularFunction&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>>(), py::arg("K"), py::arg("f"))
        .def("get_solution", &IndependentSetImprovement::get_solution)
        .def("get_fval", &IndependentSetImprovement::get_fval)
        .def("get_num_candidate_solutions", &IndependentSetImprovement::get_num_candidate_solutions)
        .def("get_num_elements_stored", &IndependentSetImprovement::get_num_elements_stored)
        .def("fit", &IndependentSetImprovement::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &IndependentSetImprovement::next, py::arg("x"));

    py::class_<SieveStreaming>(m, "SieveStreaming") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreaming::get_solution)
        .def("get_fval", &SieveStreaming::get_fval)
        .def("get_num_candidate_solutions", &SieveStreaming::get_num_candidate_solutions)
        .def("get_num_elements_stored", &SieveStreaming::get_num_elements_stored)
        .def("fit", &SieveStreaming::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &SieveStreaming::next, py::arg("x"));
    
    py::class_<SieveStreamingPP>(m, "SieveStreamingPP") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreamingPP::get_solution)
        .def("get_fval", &SieveStreamingPP::get_fval)
        .def("get_num_candidate_solutions", &SieveStreamingPP::get_num_candidate_solutions)
        .def("get_num_elements_stored", &SieveStreamingPP::get_num_elements_stored)
        .def("fit", &SieveStreamingPP::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &SieveStreamingPP::next, py::arg("x"));
    
    py::class_<ThreeSieves>(m, "ThreeSieves") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def("get_solution", &ThreeSieves::get_solution)
        .def("get_fval", &ThreeSieves::get_fval)
        .def("get_num_candidate_solutions", &ThreeSieves::get_num_candidate_solutions)
        .def("get_num_elements_stored", &ThreeSieves::get_num_elements_stored)
        .def("fit", &ThreeSieves::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &ThreeSieves::next, py::arg("x"));

    py::class_<Salsa>(m, "Salsa") 
        .def(py::init<unsigned int, SubmodularFunction&, data_t, data_t, data_t, data_t, data_t, data_t, data_t, data_t,data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("hilow_epsilon") = 0.05, py::arg("hilow_beta") = 0.1, py::arg("hilow_delta") = 0.025, py::arg("dense_beta") = 0.8, py::arg("dense_C1") = 10, py::arg("dense_C2") = 0.2, py::arg("fixed_epsilon") = 1.0/6.0)
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, data_t, data_t, data_t, data_t, data_t, data_t, data_t, data_t,data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("hilow_epsilon") = 0.05, py::arg("hilow_beta") = 0.1, py::arg("hilow_delta") = 0.025, py::arg("dense_beta") = 0.8, py::arg("dense_C1") = 10, py::arg("dense_C2") = 0.2, py::arg("fixed_epsilon") = 1.0/6.0)
        .def("get_solution", &Salsa::get_solution)
        .def("get_fval", &Salsa::get_fval)
        .def("get_num_candidate_solutions", &Salsa::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Salsa::get_num_elements_stored)
        .def("fit", &Salsa::fit, py::arg("X"), py::arg("iterations") = 1)
        .def("next", &Salsa::next, py::arg("x"));
}
