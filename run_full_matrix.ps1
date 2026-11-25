Set-Location -Path $PSScriptRoot
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONWARNINGS = 'ignore'
$jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2)
python research_runner.py --seeds 10 --jobs $jobs *> runs_full.log
python analysis/analyze.py --units runs/units --out runs/analysis
