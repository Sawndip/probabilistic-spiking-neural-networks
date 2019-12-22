add_library(pssn SHARED
            core/signal.cpp
            core/util.cpp
            generator/deterministic.cpp
           )

set_target_properties(pssn PROPERTIES VERSION ${PROJECT_VERSION})