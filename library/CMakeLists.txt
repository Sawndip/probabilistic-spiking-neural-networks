add_library(pssn SHARED
            core/signal.cpp
            core/util.cpp
            core/network.cpp
            generator/deterministic.cpp
           )

set_target_properties(pssn PROPERTIES VERSION ${PROJECT_VERSION})