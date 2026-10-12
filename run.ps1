$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv-codex\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create a Python 3.12 virtual environment and install requirements-lock.txt first. See README.md.'
}
Push-Location $projectRoot
try {
    & $projectPython app/app.py
} finally {
    Pop-Location
}
