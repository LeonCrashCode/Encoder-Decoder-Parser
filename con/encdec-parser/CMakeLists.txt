PROJECT(cnn:encdec-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(conparser nt-parser-encdec.cc oracle.cc pretrained.cc)
target_link_libraries(conparser cnn ${Boost_LIBRARIES})

