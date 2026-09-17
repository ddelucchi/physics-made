#include <iostream>
#include <stdexcept>
#include <string>

#include "physicsmade/runtime/programs.hpp"

namespace {

void printTargets() {
    std::cout << "Available launch targets:\n";
    for (const auto& target : physicsmade::runtime::availableLaunchTargets()) {
        std::cout << "  " << target.id << "\n";
        std::cout << "    " << target.summary << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            const std::string targetId = argv[1];
            return physicsmade::runtime::runLaunchTarget(targetId, std::cout);
        }

        printTargets();
        std::cout << "Enter target id: ";

        std::string selectedTarget;
        std::getline(std::cin, selectedTarget);
        if (selectedTarget.empty()) {
            return 0;
        }

        return physicsmade::runtime::runLaunchTarget(selectedTarget, std::cout);
    } catch (const std::exception& exception) {
        std::cerr << "launcher error: " << exception.what() << "\n";
        return 1;
    }
}
