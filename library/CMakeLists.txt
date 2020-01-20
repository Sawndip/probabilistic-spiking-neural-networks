add_library(pssn SHARED
            core/signal.cpp
            core/util.cpp
            core/metrics.cpp
            core/callbacks.cpp
            core/trainer.cpp
            core/network.cpp
            generator/deterministic.cpp
           )

set_target_properties(pssn PROPERTIES VERSION ${PROJECT_VERSION})

target_compile_features(pssn PUBLIC cxx_std_17)

target_compile_options(pssn PRIVATE -Wall -Wextra -pedantic -Werror)
