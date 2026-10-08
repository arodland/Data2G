function(data2g_test name)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE data2g_core)
  target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

file(GLOB DATA2G_TESTS CONFIGURE_DEPENDS test_*.cpp)
# Needs a device backend library; added below when it is built.
list(FILTER DATA2G_TESTS EXCLUDE REGEX "/test_(rig_hamlib|args|realtime|station_tx)\\.cpp$")
foreach(src ${DATA2G_TESTS})
  get_filename_component(stem ${src} NAME_WE)
  string(REGEX REPLACE "^test_" "" name ${stem})
  data2g_test(${name})
endforeach()

# Threads that must never hang: a wedge is a Timeout, not a stuck run.
set_tests_properties(audio rig PROPERTIES TIMEOUT 120)

# The libhamlib backend against Hamlib's dummy rig, directly and through a
# rigctld (model 2, the host's default path).
if(TARGET data2g_rig AND NOT WIN32)
  data2g_test(rig_hamlib)
  target_link_libraries(test_rig_hamlib PRIVATE data2g_rig)
  find_program(DATA2G_RIGCTLD rigctld HINTS "${DATA2G_HAMLIB_ROOT}/bin" NO_CACHE)
  target_compile_definitions(test_rig_hamlib PRIVATE "DATA2G_RIGCTLD=\"${DATA2G_RIGCTLD}\"")
  if(DATA2G_HAMLIB_RUNTIME_DIR)
    set_property(TARGET test_rig_hamlib APPEND PROPERTY BUILD_RPATH "${DATA2G_HAMLIB_RUNTIME_DIR}")
  endif()
  set_tests_properties(rig_hamlib PROPERTIES TIMEOUT 120)
endif()

# data2g-host's rig options (app/station.*).
if(TARGET data2g_station)
  data2g_test(args)
  target_link_libraries(test_args PRIVATE data2g_station)
  data2g_rig_rpath(test_args)
  # real-time scheduling for the audio threads (app/realtime.*)
  data2g_test(realtime)
  target_link_libraries(test_realtime PRIVATE data2g_station)
  # the transmit thread: no hole in a burst while the receive loop is stalled, and ABORT (over pipe audio)
  data2g_test(station_tx)
  target_link_libraries(test_station_tx PRIVATE data2g_station)
  data2g_rig_rpath(test_station_tx)
  set_tests_properties(station_tx PROPERTIES TIMEOUT 300)
endif()

# data2g-gui's window, offscreen, over pipe audio (no device, no display).
if(TARGET data2g_gui)
  add_executable(test_gui ${PROJECT_SOURCE_DIR}/gui/test_gui.cpp)
  target_link_libraries(test_gui PRIVATE data2g_gui)
  target_include_directories(test_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(test_gui PRIVATE "DATA2G_GUI_SHOT=\"${PROJECT_BINARY_DIR}/data2g_gui_shot.png\"")
  data2g_rig_rpath(test_gui)
  add_test(NAME gui COMMAND test_gui)
  set_tests_properties(gui PROPERTIES TIMEOUT 120 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()
