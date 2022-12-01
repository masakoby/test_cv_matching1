cmake_minimum_required(VERSION 3.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_CXX_STANDARD 17)  


project(test_cv_matching1)


file(GLOB source1 "./src/*.cpp")
set(source2 
../cv/kb_cv_matching_akaze.cpp 
../cv/kb_cv_RANSAC_calculate_transformation_matrix.cpp 
../cv/kb_cv_paste.cpp 
../cv/kb_random.cpp 
)

add_executable(test_cv_matching1 ${source1} ${source2})

include_directories("${CMAKE_CURRENT_LIST_DIR}/../common")
include_directories("${CMAKE_CURRENT_LIST_DIR}/../cv")


