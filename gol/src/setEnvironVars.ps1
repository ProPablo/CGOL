[System.Environment]::SetEnvironmentVariable('OPENCL_NVIDIA_DIR', '%CUDA_PATH%\include', [System.EnvironmentVariableTarget]::Machine)
[System.Environment]::SetEnvironmentVariable('OPENCL_NVIDIA_LIB_x64', '%CUDA_PATH%\lib\x64', [System.EnvironmentVariableTarget]::Machine)
[System.Environment]::SetEnvironmentVariable('OPENCL_NVIDIA_LIB_x86', '%CUDA_PATH%\lib\Win32', [System.EnvironmentVariableTarget]::Machine)
cmd /c pause | out-null;