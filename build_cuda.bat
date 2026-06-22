@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
cmake -S . -B build_cuda -G "NMake Makefiles" -DTURBO_CPP_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="%CUDACXX%"
if errorlevel 1 exit /b %errorlevel%
cmake --build build_cuda --target turbo_cpp_tests turbo_cpp_bench
if errorlevel 1 exit /b %errorlevel%
build_cuda\turbo_cpp_tests.exe
