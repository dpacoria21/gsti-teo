$ErrorActionPreference = 'Stop'
$taskState = Join-Path $PSScriptRoot '.runtime\server.json'
$taskApp = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path -LiteralPath $taskState)) { Write-Host 'No hay un servidor registrado para este proyecto.'; return }
$taskServer = Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskServer.pid)" -ErrorAction SilentlyContinue
if ($taskProcess -and $taskProcess.CommandLine -like "*$taskApp*") {
    Stop-Process -Id $taskServer.pid
    Write-Host 'Servidor del mapa detenido.'
} else { Write-Host 'El proceso registrado ya no corresponde a esta aplicación.' }

