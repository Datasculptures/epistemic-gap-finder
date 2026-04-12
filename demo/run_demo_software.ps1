# EGF Software Domain Demo
# Runs the full pipeline against the datasculptures portfolio.
# From the project root:
#   .venv\Scripts\Activate.ps1
#   .\demo\run_demo_software.ps1

$ErrorActionPreference = "Stop"

$inputDir  = Join-Path $PSScriptRoot "software"
$outputDir = Join-Path $PSScriptRoot ".." "egf_output_software"

Write-Host "EGF Software Demo" -ForegroundColor Cyan
Write-Host "Input:  $inputDir"
Write-Host "Output: $outputDir"
Write-Host ""

egf analyse $inputDir `
    --output $outputDir `
    --domain software-tool `
    --isolation-min 0.2 `
    --max-gaps 5 `
    --open

Write-Host ""
Write-Host "Done. Report: $outputDir\report.html" -ForegroundColor Green
