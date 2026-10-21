$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        & uv venv --python 3.12 (Join-Path $taskRoot '.venv')
    } else {
        & python -m venv (Join-Path $taskRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno Python. Se necesita Python 3.12.' }
}
$taskRequirements = Join-Path $taskRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $taskRequirements)) { $taskRequirements = Join-Path $taskRoot 'requirements.in' }
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv pip install --python $taskPython -r $taskRequirements
} else {
    & $taskPython -m ensurepip --upgrade
    & $taskPython -m pip install -r $taskRequirements
}
if ($LASTEXITCODE -ne 0) { throw 'La instalación no terminó. Revise la conexión y el mensaje anterior.' }
Write-Host 'Entorno listo. Ejecute INICIAR.cmd para abrir el mapa.'

