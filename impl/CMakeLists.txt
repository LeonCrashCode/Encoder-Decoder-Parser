CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET
	greedy
	greedy_attention
	greedy_attention_s
	greedy_attention_b
	greedy_attention_sb
	greedy_attention_sb_cond
	greedy_attention_sb_cond_hybrid
	greedy_attention_sb_cond_hybrid_dy
	greedy_attention_sb_cond2
	greedy_attention_sb_noC
	greedy_attention_sb_pst

	greedy_attention_sb_tree
	greedy_attention_sb_tree_label
	greedy_attention_sbd_tree_label
	greedy_attention_sb_tree2

###condition
	greedy_sb
	greedy_attention_sb_s0b0
	greedy_sb_conv

	greedy_localattention_sb_cond
	cuda_att_sb

###
	att_sb

###beam
	greedy_attention_sb_beam
	)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS})
    add_dependencies(${TARGET} dynetcuda)
    target_link_libraries(${TARGET} dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()


#ADD_EXECUTABLE(nt-parser-encdec nt-parser-encdec.cc oracle.cc pretrained.cc)
#target_link_libraries(nt-parser-encdec dynet ${LIBS})

