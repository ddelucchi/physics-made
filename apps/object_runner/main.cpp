#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "physicsmade/physics/kinematics_model.hpp"
#include "physicsmade/runtime/programs.hpp"

namespace {

void printUsage() {
    std::cout << "Usage: physicsmade_object_runner [--list] [--object <id>] [--regime <nonrelativistic|special-relativistic>] [--integrator <semi-implicit-euler|velocity-verlet|runge-kutta-4>]\n";
}

physicsmade::physics::KinematicRegime parseRegime(std::string_view value) {
    if (value == "nonrelativistic") {
        return physicsmade::physics::KinematicRegime::NonRelativistic;
    }

    if (value == "special-relativistic") {
        return physicsmade::physics::KinematicRegime::SpecialRelativistic;
    }

    throw std::runtime_error("unknown regime");
}

void printPrograms() {
    std::cout << "Available object programs:\n";
    for (const auto& program : physicsmade::runtime::availableObjectPrograms()) {
        std::cout << "  " << program.id << " [default regime: " << physicsmade::physics::toString(program.defaultRegime) << "]\n";
        std::cout << "    " << program.summary << "\n";
    }

    std::cout << "Available integrators:\n";
    for (const auto& integrator : physicsmade::runtime::availableIntegrators()) {
        std::cout << "  " << integrator.id << "\n";
        std::cout << "    " << integrator.summary << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string objectId;
        bool listOnly = false;
        auto regime = physicsmade::physics::KinematicRegime::NonRelativistic;
        std::string integratorId{"velocity-verlet"};

        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--list") {
                listOnly = true;
                continue;
            }

            if (argument == "--object" && (index + 1) < argc) {
                objectId = argv[++index];
                continue;
            }

            if (argument == "--regime" && (index + 1) < argc) {
                regime = parseRegime(argv[++index]);
                continue;
            }

            if (argument == "--integrator" && (index + 1) < argc) {
                integratorId = argv[++index];
                continue;
            }

            printUsage();
            return 1;
        }

        if (listOnly || objectId.empty()) {
            printPrograms();
            if (objectId.empty()) {
                return 0;
            }
        }

        return physicsmade::runtime::runObjectProgram(objectId, regime, integratorId, std::cout);
    } catch (const std::exception& exception) {
        std::cerr << "object runner error: " << exception.what() << "\n";
        return 1;
    }
}
