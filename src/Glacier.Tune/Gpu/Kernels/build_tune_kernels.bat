@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Glacier.Tune CUDA Kernel Fatbinary Builder
echo ===================================================

where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    for %%C in (
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin"
    ) do (
        if exist "%%~C\nvcc.exe" (
            set "PATH=%%~C;!PATH!"
            goto :FoundNvcc
        )
    )
    echo [ERROR] nvcc.exe not found.
    exit /b 1
)
:FoundNvcc

where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set "VS_VCVARS="
    for %%V in (
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if exist %%V (
            set "VS_VCVARS=%%~V"
            goto :FoundVcVars
        )
    )

    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "%VSWHERE%" (
        for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -property installationPath`) do (
            if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" (
                set "VS_VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
                goto :FoundVcVars
            )
        )
    )

:FoundVcVars
    if defined VS_VCVARS (
        call "!VS_VCVARS!" >nul
    )
)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo [INFO] Compiling tune_bwd.cu to tune_bwd.cubin...
nvcc -fatbin -O3 ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_80,code=sm_80 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_90,code=sm_90 ^
  -gencode arch=compute_75,code=compute_75 ^
  -o tune_bwd.cubin tune_bwd.cu

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Successfully built tune_bwd.cubin fatbinary.
    for %%F in (tune_bwd.cubin) do echo [INFO] Size: %%~zF bytes
) else (
    echo [ERROR] nvcc compilation failed with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)
