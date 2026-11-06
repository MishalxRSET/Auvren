# Install Piper (offline text-to-speech) and one voice for Auvren.
#
#   powershell -ExecutionPolicy Bypass -File scripts\get_piper.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\get_piper.ps1 -Voice en_US-ryan-high
#
# Puts piper.exe under <data>\piper\piper\ and the voice under
# <data>\piper\voices\, which is where tts.py looks when PIPER_PATH and
# PIPER_MODEL are unset (<data> is JARVIS_DATA_DIR, else .\data). Then speaks
# one test sentence to <data>\piper\test.wav and plays it, so you know it works
# before starting the server.
#
# Voices: https://huggingface.co/rhasspy/piper-voices (the catalogue is
# voices.json there). There are no Indian English (en_IN) voices; en_GB-alan
# is a British male voice, the nearest to the butler. More than one installed
# voice means setting PIPER_MODEL in .env to choose.

param(
    [string]$Voice = "en_GB-alan-medium",
    [string]$PiperRelease = "2023.11.14-2"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x slower with the bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# A voice name becomes part of a URL and a file name: accept only the
# catalogue's own shape, lang_REGION-name-quality.
if ($Voice -notmatch '^[a-z]{2,3}_[A-Z]{2}-[A-Za-z0-9_]+-(x_low|low|medium|high)$') {
    throw "'$Voice' is not a Piper voice name (expected e.g. en_GB-alan-medium)."
}

$repo = Split-Path -Parent $PSScriptRoot
$data = if ($env:JARVIS_DATA_DIR) { $env:JARVIS_DATA_DIR } else { Join-Path $repo "data" }
$root = Join-Path $data "piper"
$bin = Join-Path $root "piper"
$voices = Join-Path $root "voices"
New-Item -ItemType Directory -Force -Path $root, $voices | Out-Null

# --- Piper itself ----------------------------------------------------------
$exe = Join-Path $bin "piper.exe"
if (Test-Path $exe) {
    Write-Host "Piper is already installed: $exe"
} else {
    $url = "https://github.com/rhasspy/piper/releases/download/$PiperRelease/piper_windows_amd64.zip"
    $zip = Join-Path $env:TEMP "piper_windows_amd64.zip"
    $unpack = Join-Path $env:TEMP ("piper-unpack-" + [guid]::NewGuid())
    Write-Host "Downloading Piper $PiperRelease ..."
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $unpack -Force
    $found = Get-ChildItem -Path $unpack -Recurse -Filter "piper.exe" | Select-Object -First 1
    if (-not $found) { throw "The Piper download did not contain piper.exe." }
    # Keep piper.exe together with espeak-ng-data and its DLLs.
    if (Test-Path $bin) { Remove-Item -Recurse -Force $bin }
    Move-Item -Path $found.DirectoryName -Destination $bin
    Remove-Item -Force $zip
    Remove-Item -Recurse -Force $unpack -ErrorAction SilentlyContinue
    Write-Host "Installed Piper: $exe"
}

# --- the voice -------------------------------------------------------------
$lang, $name, $quality = $Voice -split '-', 3
$family = $lang.Split('_')[0]
$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$family/$lang/$name/$quality/$Voice"
$model = Join-Path $voices "$Voice.onnx"
foreach ($ext in @(".onnx", ".onnx.json")) {
    $dest = Join-Path $voices "$Voice$ext"
    if (Test-Path $dest) { continue }
    Write-Host "Downloading $Voice$ext ..."
    Invoke-WebRequest -UseBasicParsing -Uri "$base$ext" -OutFile $dest
}

# --- prove it speaks ---------------------------------------------------------
$test = Join-Path $root "test.wav"
Write-Host "Speaking a test sentence ..."
"Good evening. Auvren's voice is working." | & $exe --model $model --output_file $test
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $test)) {
    throw "Piper did not produce audio (exit $LASTEXITCODE)."
}
try { (New-Object Media.SoundPlayer $test).PlaySync() } catch { }

$installed = @(Get-ChildItem -Path $voices -Filter "*.onnx").Count
Write-Host ""
Write-Host "Done. Piper wrote $test."
if ($installed -gt 1) {
    Write-Host "You have $installed voices installed, so choose one in .env:"
    Write-Host "  PIPER_MODEL=$model"
} else {
    Write-Host "Auvren will find it on its own; no .env change is needed."
}
