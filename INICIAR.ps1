param([int]$Port = 8501, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskApp = Join-Path $taskRoot 'app.py'
$taskRuntime = Join-Path $taskRoot '.runtime'
$taskState = Join-Path $taskRuntime 'server.json'
$taskUrl = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Primero ejecute PREPARAR.ps1 para instalar las dependencias.' }
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
if (Test-Path -LiteralPath $taskState) {
    $taskPrevious = Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
    $taskExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrevious.pid)" -ErrorAction SilentlyContinue
    if ($taskExistingProcess -and $taskExistingProcess.CommandLine -like "*$taskApp*" -and $taskPrevious.port -eq $Port) {
        try {
            $taskResponse = Invoke-WebRequest "$taskUrl/_stcore/health" -TimeoutSec 3 -UseBasicParsing
            if ($taskResponse.StatusCode -eq 200) {
                Write-Host "Mapa disponible: $taskUrl"
                if (-not $NoBrowser) { Start-Process $taskUrl }
                return
            }
        } catch { }
    }
}
$taskListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($taskListener) { throw "El puerto $Port está ocupado. Use INICIAR.ps1 -Port 8502." }
$taskArgs = @('-m', 'streamlit', 'run', "`"$taskApp`"", '--server.address=127.0.0.1', "--server.port=$Port", '--server.headless=true', '--browser.gatherUsageStats=false')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuntime 'server.log') -RedirectStandardError (Join-Path $taskRuntime 'server-error.log')
@{pid=$taskProcess.Id;port=$Port;project=$taskRoot;url=$taskUrl} | ConvertTo-Json | Set-Content -LiteralPath $taskState -Encoding UTF8
for ($taskAttempt=0; $taskAttempt -lt 50; $taskAttempt++) {
    Start-Sleep -Milliseconds 300
    if ($taskProcess.HasExited) { throw 'El servidor no pudo iniciar. Revise .runtime/server-error.log.' }
    try {
        $taskResponse = Invoke-WebRequest "$taskUrl/_stcore/health" -TimeoutSec 2 -UseBasicParsing
        if ($taskResponse.StatusCode -eq 200) {
            Write-Host "Mapa disponible: $taskUrl"
            if (-not $NoBrowser) { Start-Process $taskUrl }
            return
        }
    } catch { }
}
throw 'El servidor está tardando en arrancar. Revise .runtime/server-error.log.'

