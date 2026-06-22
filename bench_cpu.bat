@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
cmake --build build --target turbo_cpp_bench
if errorlevel 1 exit /b %errorlevel%
build\turbo_cpp_bench.exe
