# Run Auvren in the background from Windows sign-in, listening for "Hey Auvren".
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1            # install + start now
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -NoStart   # install only
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove    # uninstall
#
# What it does:
#   1. Builds the page once (frontend\dist), so the server can show it without
#      the Vite dev server.
#   2. Puts a shortcut in your Startup folder that runs
#        pythonw.exe server.py --background
#      -- pythonw, so there is no console window. --background serves the page
#      over plain HTTP on 127.0.0.1 (a loopback address is a secure context, so
#      no certificate warning) and logs to <data>\logs\auvren.log.
#   3. Starts it now.
#
# When you say "Hey Auvren" and no Auvren window is open, the server opens one
# (an app window of Chrome, or of Edge, which every Windows has), plays the
# intro, and asks what you want. That needs speech recognition on the laptop
# (requirements-stt.txt) -- the browser's recogniser cannot hear you while no
# page is open.

param(
    [switch]$Remove,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath("Startup")
$link = Join-Path $startup "Auvren.lnk"

if ($Remove) {
    if (Test-Path $link) {
        Remove-Item -Force $link
        Write-Host "Removed $link. Auvren will no longer start with Windows."
    } else {
        Write-Host "Auvren was not set to start with Windows."
    }
    # Stop a running background server, if this repo's server.py is one.
    Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
        Where-Object { $_.CommandLine -like "*server.py*--background*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "Stopped the running Auvren (pid $($_.ProcessId))." }
    exit 0
}

# --- pythonw ------------------------------------------------------------------
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "python is not on PATH." }
$pythonw = Join-Path (Split-Path -Parent $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe is not next to $python." }

# --- the page ----------------------------------------------------------------
$dist = Join-Path $repo "frontend\dist\index.html"
if (-not (Test-Path $dist)) {
    Write-Host "Building the page (frontend\dist) ..."
    Push-Location (Join-Path $repo "frontend")
    try {
        npm install
        npm run build
    } finally { Pop-Location }
}

# --- speech recognition, without which nothing can wake it -------------------
& $python -c "import stt, sys; p = stt.engine_problem(); print(p or 'ok'); sys.exit(1 if p else 0)" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning ("On-device speech recognition is not set up, so nothing can hear " +
                   "'Hey Auvren' while no window is open. Run:`n" +
                   "  pip install -r requirements-stt.txt`n  python stt.py --setup")
}

# --- the shortcut --------------------------------------------------------------
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = $pythonw
$sc.Arguments = '"' + (Join-Path $repo "server.py") + '" --background'
$sc.WorkingDirectory = $repo
$sc.Description = "Auvren voice assistant (background)"
$sc.WindowStyle = 7
$sc.Save()
Write-Host "Auvren will start in the background when you sign in ($link)."

if (-not $NoStart) {
    Start-Process -FilePath $pythonw -ArgumentList $sc.Arguments -WorkingDirectory $repo -WindowStyle Hidden
    Write-Host "Started. Say ""Hey Auvren""."
}
