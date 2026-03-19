cmake_minimum_required(VERSION 3.16)

# Try Qt6 first, fallback to Qt5
find_package(Qt6 QUIET COMPONENTS Core Gui Widgets Network)
if(Qt6_FOUND)
    set(QT_VERSION_MAJOR 6)
    set(QT_LIBRARIES Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Network)
    find_package(Qt6LinguistTools QUIET)
else()
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Network)
    set(QT_VERSION_MAJOR 5)
    set(QT_LIBRARIES Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Network)
    find_package(Qt5LinguistTools QUIET)
endif()

pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(VINPUT_GUI_SOURCES
    main.cpp
    mainwindow.cpp
    ${CMAKE_SOURCE_DIR}/src/common/pipewire_device.cpp
)

set(VINPUT_GUI_HEADERS
    mainwindow.h
)

add_executable(vinput-gui
    ${VINPUT_GUI_SOURCES}
    ${VINPUT_GUI_HEADERS}
)

if(Qt6LinguistTools_FOUND)
    set(VINPUT_GUI_TS
        ${CMAKE_SOURCE_DIR}/i18n/vinput-gui_zh_CN.ts
    )
    qt6_add_translation(VINPUT_GUI_QM ${VINPUT_GUI_TS})
    add_custom_target(vinput-gui-translations ALL DEPENDS ${VINPUT_GUI_QM})
elseif(Qt5LinguistTools_FOUND)
    set(VINPUT_GUI_TS
        ${CMAKE_SOURCE_DIR}/i18n/vinput-gui_zh_CN.ts
    )
    qt5_add_translation(VINPUT_GUI_QM ${VINPUT_GUI_TS})
    add_custom_target(vinput-gui-translations ALL DEPENDS ${VINPUT_GUI_QM})
endif()

target_include_directories(vinput-gui PRIVATE
    ${PIPEWIRE_INCLUDE_DIRS}
)

target_link_libraries(vinput-gui
    PRIVATE
    ${QT_LIBRARIES}
    vinput-common
    ${PIPEWIRE_LIBRARIES}
)

install(TARGETS vinput-gui
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(Qt6LinguistTools_FOUND OR Qt5LinguistTools_FOUND)
    install(FILES ${VINPUT_GUI_QM}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/fcitx5-vinput/i18n
    )
endif()
