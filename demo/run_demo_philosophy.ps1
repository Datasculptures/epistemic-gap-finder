# EGF Philosophy Domain Demo
# Runs the full pipeline against ten schools of Western philosophy.
# From the project root:
#   .venv\Scripts\Activate.ps1
#   .\demo\run_demo_philosophy.ps1

$ErrorActionPreference = "Stop"

$inputDir  = Join-Path $PSScriptRoot "philosophy"
$outputDir = Join-Path $PSScriptRoot ".." "egf_output_philosophy"

Write-Host "EGF Philosophy Demo" -ForegroundColor Cyan
Write-Host "Input:  $inputDir"
Write-Host "Output: $outputDir"
Write-Host ""

egf analyse $inputDir `
    --output $outputDir `
    --domain philosophy `
    --isolation-min 0.15 `
    --max-gaps 5 `
    --open

Write-Host ""
Write-Host "Done. Report: $outputDir\report.html" -ForegroundColor Green
