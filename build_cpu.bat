@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
cmake -S . -B build -G "NMake Makefiles" -DTURBO_CPP_ENABLE_CUDA=OFF
if errorlevel 1 exit /b %errorlevel%
cmake --build build --target turbo_cpp_tests
if errorlevel 1 exit /b %errorlevel%
build\turbo_cpp_tests.exe
