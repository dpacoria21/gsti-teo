param([ValidateSet('auto','cpu','cuda')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv-ml\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        & uv venv --python 3.12 (Join-Path $taskRoot '.venv-ml')
    } else {
        & python -m venv (Join-Path $taskRoot '.venv-ml')
    }
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno IA. Se necesita Python 3.12.' }
}
$taskDevice = $Device
if ($taskDevice -eq 'auto') {
    $taskDevice = 'cpu'
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $taskDevice = 'cuda' }
}
$taskIndex = 'https://download.pytorch.org/whl/cpu'
if ($taskDevice -eq 'cuda') { $taskIndex = 'https://download.pytorch.org/whl/cu128' }
Write-Host "Preparando IA con $taskDevice. La primera descarga de PyTorch puede ser grande."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv pip install --python $taskPython 'torch==2.11.0' 'torchvision==0.26.0' --index $taskIndex
} else {
    & $taskPython -m ensurepip --upgrade
    & $taskPython -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url $taskIndex
}
if ($LASTEXITCODE -ne 0) { throw 'Falló PyTorch. En un equipo sin CUDA pruebe PREPARAR_IA.ps1 -Device cpu.' }
# Preserve the installed CPU/CUDA wheels; install pinned support libraries separately.
$taskSupport = Join-Path $taskRoot '.runtime\requirements-ml-support.txt'
New-Item -ItemType Directory -Path (Split-Path -Parent $taskSupport) -Force | Out-Null
Get-Content -LiteralPath (Join-Path $taskRoot 'requirements-ml.lock.txt') | Where-Object { $_ -notmatch '^torch(vision)?==' } | Set-Content -LiteralPath $taskSupport -Encoding UTF8
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv pip install --python $taskPython -r $taskSupport
} else {
    & $taskPython -m pip install -r $taskSupport
}
if ($LASTEXITCODE -ne 0) { throw 'Falló la instalación de las bibliotecas geográficas.' }
& $taskPython -c 'from src.building_detection import ensure_model; print(ensure_model())'
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron verificar los pesos preentrenados.' }
Write-Host 'Modelo y entorno IA listos. Abra INICIAR.cmd y use Detección desde imagen.'
