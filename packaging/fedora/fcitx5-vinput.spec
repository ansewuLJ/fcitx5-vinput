Name:           fcitx5-vinput
Version:        %{?_vinput_version}%{!?_vinput_version:0.1.6}
Release:        1%{?dist}
Summary:        Offline voice input addon for Fcitx5
License:        GPL-3.0-only
URL:            https://github.com/xifan2333/fcitx5-vinput
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz

BuildRequires:  cmake >= 3.16
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  pkgconfig
BuildRequires:  gettext
BuildRequires:  cmake(Fcitx5Core)
BuildRequires:  cmake(Fcitx5Config)
BuildRequires:  cmake(nlohmann_json) >= 3.2.0
BuildRequires:  cmake(Qt5Core)
BuildRequires:  cmake(Qt5Gui)
BuildRequires:  cmake(Qt5Widgets)
BuildRequires:  cmake(Qt5LinguistTools)
BuildRequires:  pkgconfig(libcurl)
BuildRequires:  pkgconfig(openssl)
BuildRequires:  pkgconfig(libarchive)
BuildRequires:  pkgconfig(libpipewire-0.3)
BuildRequires:  pkgconfig(libsystemd)
BuildRequires:  cli11-devel

Requires:       fcitx5
Requires:       pipewire
Requires:       curl
Requires:       systemd

%description
Local offline voice input plugin for Fcitx5, powered by sherpa-onnx
for on-device speech recognition with optional LLM post-processing
via any OpenAI-compatible API.

%prep
%autosetup -n %{name}-%{version}

%build
%cmake -G Ninja \
    -DVINPUT_PROJECT_VERSION=%{version} \
    -DVINPUT_PACKAGE_RELEASE=%{release} \
    -DVINPUT_PACKAGE_HOMEPAGE_URL=%{url}
%cmake_build

%install
%cmake_install

%files
%license LICENSE
%{_bindir}/vinput
%{_bindir}/vinput-daemon
%{_bindir}/vinput-gui
%{_libdir}/fcitx5/vinput.so
%{_libdir}/fcitx5-vinput/
%{_datadir}/fcitx5/addon/vinput.conf
%{_datadir}/fcitx5/inputmethod/vinput.conf
%{_datadir}/dbus-1/services/org.fcitx.Vinput.service
%{_datadir}/systemd/user/vinput-daemon.service
%{_datadir}/locale/*/LC_MESSAGES/fcitx5-vinput.mo
%{_datadir}/fcitx5-vinput/

%changelog
* Tue Mar 18 2026 xifan2333 <noreply@github.com> - 0.1.6-1
- Initial RPM package
