"""Build the Ramulator probe from a verified archive using CMake and Ninja."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def command(*args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("work", type=Path, help="New directory for extracted sources and build")
    parser.add_argument("--compiler", help="Path to a C++20 compiler or compiler wrapper")
    parser.add_argument("--llmsimulator", action="store_true", help="Build the coupled LLMSimulator instead of the probe")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir()
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read("SUBMISSION_MANIFEST.json"))["files"]
        for name, expected in manifest.items():
            if name.startswith(("src/", "patch/", "eval/")) or name in {
                "CMakeLists.txt", "config.yaml", "dram_config_HBM3_80GB.yaml",
                "dram_config_HBM3E_192GB.yaml",
            } or name.endswith(
                "ramulator-gcc15-cstdint.patch"
            ):
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Archive source hash mismatch: {name}")
                archive.extract(name, work)
    source = work / "src/dram/ramulator2"
    command("git", "apply", work / "patch/ramulator2_pim.patch", cwd=source)
    command("git", "apply", work / "midsem/ramulator-gcc15-cstdint.patch", cwd=source)
    probe = Path(__file__).resolve().with_name("dram_probe.cpp").as_posix()
    # The PIM patch converts upstream Ramulator into LLMSimulator subtargets.
    # Object linkage retains static factory registrations in the standalone probe.
    if not args.llmsimulator:
        write_probe_cmake(work, probe)
    adapt_sources(source)
    configure(args, work, source)


def write_probe_cmake(work, probe):
    (work / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.14)\nproject(probe LANGUAGES CXX)\n'
        'include_directories(src)\nadd_compile_definitions(FMT_CONSTEVAL=)\n'
        'set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n'
        'add_library(dram OBJECT unused.cpp)\nadd_library(hardware OBJECT unused.cpp)\n'
        'add_library(llm_system OBJECT unused.cpp)\n'
        'add_subdirectory(src/dram/ramulator2)\n'
        f'add_executable(dram_probe "{probe}")\n'
        'target_link_libraries(dram_probe PRIVATE dram)\n'
        'get_target_property(parts dram LINK_LIBRARIES)\n'
        'foreach(part IN LISTS parts)\n'
        '  if(TARGET ${part})\n'
        '    get_target_property(kind ${part} TYPE)\n'
        '    if(kind STREQUAL "OBJECT_LIBRARY")\n'
        '      target_sources(dram_probe PRIVATE $<TARGET_OBJECTS:${part}>)\n'
        '    endif()\n'
        '  endif()\n'
        'endforeach()\n', encoding="utf-8"
    )
    (work / "unused.cpp").touch()


def adapt_sources(source):
    # Port two POSIX assumptions in the archived code to Windows.
    types = source / "src/base/type.h"
    types.write_text("#if defined(_WIN32)\nusing uint = unsigned int;\n#endif\n" +
                     types.read_text(encoding="utf-8"), encoding="utf-8")
    config = source / "src/base/config.cpp"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "YAML::LoadFile(path)", "YAML::LoadFile(path.string())"), encoding="utf-8")
    for filename in ("trace_recorder.cpp", "cmd_counter.cpp"):
        plugin = source / "src/dram_controller/impl/plugin" / filename
        contents = plugin.read_text(encoding="utf-8").replace(
            'param<std::string>("path").desc("Path to the trace file").required();',
            'std::filesystem::path(std::string(param<std::string>("path").desc('
            '"Path to the trace file").required()));')
        plugin.write_text(contents, encoding="utf-8")
    # PE/COFF may initialize an implementation before its interface. Register
    # implementation metadata in an empty interface slot and fill its name later.
    factory = source / "src/base/factory.cpp"
    contents = factory.read_text(encoding="utf-8").replace(
        'if (auto it = m_registry.find(ifce_name); it == m_registry.end()) {\n'
        '    m_registry[ifce_name] = {ifce_name, ifce_desc};',
        'if (auto& entry = m_registry[ifce_name]; entry.name.empty()) {\n'
        '    entry.name = ifce_name;\n    entry.desc = ifce_desc;')
    marker = "  // First we search for the interface metadata."
    contents = contents.replace(marker, "  m_registry.try_emplace(ifce_name);\n" + marker)
    factory.write_text(contents, encoding="utf-8")
    registry = source / "src/base/factory.h"
    registry.write_text(registry.read_text(encoding="utf-8").replace(
        "inline static Registry_t<InterfaceInfo> m_registry;",
        "static Registry_t<InterfaceInfo>& registry() {"
        " static Registry_t<InterfaceInfo> value; return value; }"),
        encoding="utf-8")
    factory.write_text(factory.read_text(encoding="utf-8").replace(
        "m_registry", "registry()"), encoding="utf-8")


def configure(args, work, source):
    options = ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
    for dependency in ("yaml-cpp", "spdlog", "argparse"):
        options.append(f"-DFETCHCONTENT_SOURCE_DIR_{dependency.upper()}={source / 'ext' / dependency}")
    if args.compiler:
        options.append(f"-DCMAKE_CXX_COMPILER={args.compiler}")
    target = "run" if args.llmsimulator else "dram_probe"
    command("cmake", "-S", work, "-B", work / "build", "-G", "Ninja", *options)
    command("cmake", "--build", work / "build", "--target", target,
            "--parallel", args.jobs)
    print(work / "build" / (target + ".exe" if os.name == "nt" else target))


if __name__ == "__main__":
    main()
