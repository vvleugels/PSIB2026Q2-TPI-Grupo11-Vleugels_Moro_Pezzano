$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $projectRoot 'PSIB.bat'
$icon = Join-Path $projectRoot 'assets\icon\psib.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell

foreach ($shortcutPath in @((Join-Path $projectRoot 'PSIB.lnk'), (Join-Path $desktop 'PSIB.lnk'))) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $target
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.IconLocation = "$icon,0"
    $shortcut.WindowStyle = 7  # Minimized.
    $shortcut.Description = 'Launch the PSIB liver tumor GUI'
    $shortcut.Save()
    Write-Output "Created $shortcutPath"
}

if (-not (Test-Path -LiteralPath $icon)) {
    Write-Warning "Shortcut created, but its icon is missing. Run python scripts/build_icons.py to generate the icon assets from the supplied artwork."
}
