[CmdletBinding()]
param(
    [string]$IconPath = "images\\snipper-icon.png",
    [string]$SplashPath = "images\\snipper-splash.png",
    [string]$TesseractRoot = "",
    [string]$AppName = "DataSnipper",
    [string]$VersionFile = "VERSION",
    [string]$CompanyName = "Data Snipper",
    [string]$WixPath = "",
    [switch]$OneFile,
    [switch]$OneDirSfx
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $null
    }
    $resolved = Resolve-Path -Path $PathValue -ErrorAction SilentlyContinue
    if (-not $resolved) {
        $candidate = Join-Path $PSScriptRoot $PathValue
        $resolved = Resolve-Path -Path $candidate -ErrorAction SilentlyContinue
    }
    return $resolved
}

function Find-TesseractRoot([string]$RootOverride) {
    if (-not [string]::IsNullOrWhiteSpace($RootOverride)) {
        return $RootOverride
    }

    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles "Tesseract-OCR"
    }
    if ($env:ProgramFiles -and $env:ProgramFiles -ne "C:\\Program Files") {
        $candidates += Join-Path "C:\\Program Files" "Tesseract-OCR"
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates += Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR"
    }
    if ($env:LocalAppData) {
        $candidates += Join-Path $env:LocalAppData "Programs\\Tesseract-OCR"
    }

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "tesseract.exe")) {
            return $candidate
        }
    }

    return $null
}

Set-Location $PSScriptRoot

$buildSingleFile = $OneFile
$buildSfx = $OneDirSfx
if (-not $buildSingleFile -and -not $buildSfx) {
    $buildSingleFile = $true
    $buildSfx = $true
}

$iconResolved = Resolve-ProjectPath $IconPath
if (-not $iconResolved) {
    throw "Icon file not found. Provide -IconPath or create $IconPath"
}

$splashResolved = Resolve-ProjectPath $SplashPath
if (-not $splashResolved) {
    throw "Splash file not found. Provide -SplashPath or create $SplashPath"
}

$tessRoot = Find-TesseractRoot $TesseractRoot
if (-not $tessRoot) {
    throw "Tesseract root not found. Provide -TesseractRoot (folder containing tesseract.exe)."
}

$tessExe = Join-Path $tessRoot "tesseract.exe"
if (-not (Test-Path $tessExe)) {
    throw "tesseract.exe not found under $tessRoot"
}

$tessData = Join-Path $tessRoot "tessdata"
if (-not (Test-Path $tessData)) {
    throw "tessdata folder not found under $tessRoot"
}

$pyiVersion = & python -m PyInstaller --version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller not available. Install with: pip install pyinstaller"
}

$buildDir = Join-Path $PSScriptRoot "build"
New-Item -Path $buildDir -ItemType Directory -Force | Out-Null

$hookPath = Join-Path $buildDir "pyi_hook_tesseract.py"
$hookLines = @(
    'import os',
    'import sys',
    '',
    'root = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))',
    'tess_root = os.path.join(root, "Tesseract-OCR")',
    'tess_exe = os.path.join(tess_root, "tesseract.exe")',
    'tess_data = os.path.join(tess_root, "tessdata")',
    'if os.path.isfile(tess_exe):',
    '    os.environ.setdefault("TESSERACT_CMD", tess_exe)',
    '    if os.path.isdir(tess_data):',
    '        os.environ.setdefault("TESSDATA_PREFIX", tess_data)'
)
$hookLines | Set-Content -Path $hookPath -Encoding ASCII

function New-IconForExe([string]$SourceIconPath) {
    $ext = [IO.Path]::GetExtension($SourceIconPath).ToLowerInvariant()
    if ($ext -eq ".ico") {
        return $SourceIconPath
    }

    $iconOut = Join-Path $buildDir "app.ico"
    $python = @"
from PIL import Image
img = Image.open(r"$SourceIconPath")
img.save(r"$iconOut", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
"@
    $python | python -
    if (-not (Test-Path $iconOut)) {
        throw "Failed to create ICO from $SourceIconPath. Ensure Pillow is installed."
    }
    return $iconOut
}

$iconForExe = New-IconForExe $iconResolved.Path
$iconRuntimePath = Join-Path $buildDir "snipper-icon.ico"
Copy-Item -Path $iconForExe -Destination $iconRuntimePath -Force

function New-BmpFromImage([string]$SourceImagePath, [int]$Width, [int]$Height, [string]$OutName, [switch]$Crop) {
    $bmpOut = Join-Path $buildDir $OutName
    $python = @"
from PIL import Image
img = Image.open(r"$SourceImagePath")
target_w, target_h = $Width, $Height
if $($Crop.IsPresent):
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, int((new_w - target_w) / 2))
    top = max(0, int((new_h - target_h) / 2))
    img = img.crop((left, top, left + target_w, top + target_h))
else:
    img = img.resize((target_w, target_h), Image.LANCZOS)
img.save(r"$bmpOut", format="BMP")
"@
    $python | python -
    if (-not (Test-Path $bmpOut)) {
        throw "Failed to create BMP from $SourceImagePath. Ensure Pillow is installed."
    }
    return $bmpOut
}

function New-BannerBmp([string]$SourceImagePath, [int]$Width, [int]$Height, [int]$PadLeft, [string]$OutName) {
    $bmpOut = Join-Path $buildDir $OutName
    $python = @"
from PIL import Image, ImageColor
img = Image.open(r"$SourceImagePath")
target_w, target_h = $Width, $Height
pad_left = $PadLeft
bg = Image.new("RGB", (target_w, target_h), ImageColor.getrgb("#f4f4f4"))
img_w, img_h = img.size
scale = target_h / img_h
new_w = int(round(img_w * scale))
new_h = target_h
img = img.resize((new_w, new_h), Image.LANCZOS)
max_w = max(1, target_w - pad_left)
if new_w > max_w:
    left = max(0, int((new_w - max_w) / 2))
    img = img.crop((left, 0, left + max_w, new_h))
paste_x = target_w - img.size[0]
bg.paste(img, (paste_x, 0))
bg.save(r"$bmpOut", format="BMP")
"@
    $python | python -
    if (-not (Test-Path $bmpOut)) {
        throw "Failed to create banner BMP from $SourceImagePath. Ensure Pillow is installed."
    }
    return $bmpOut
}

function New-DialogBmp([string]$SourceImagePath, [int]$Width, [int]$Height, [int]$ImageWidth, [string]$OutName) {
    $bmpOut = Join-Path $buildDir $OutName
    $python = @"
from PIL import Image, ImageColor
img = Image.open(r"$SourceImagePath")
target_w, target_h = $Width, $Height
image_w = $ImageWidth
bg = Image.new("RGB", (target_w, target_h), ImageColor.getrgb("#f4f4f4"))
img_w, img_h = img.size
scale = target_h / img_h
new_w = int(round(img_w * scale))
new_h = target_h
img = img.resize((new_w, new_h), Image.LANCZOS)
if new_w > image_w:
    left = max(0, int((new_w - image_w) / 2))
    img = img.crop((left, 0, left + image_w, new_h))
else:
    pad_x = int((image_w - new_w) / 2)
    strip = Image.new("RGB", (image_w, target_h), ImageColor.getrgb("#f4f4f4"))
    strip.paste(img, (pad_x, 0))
    img = strip
bg.paste(img, (0, 0))
bg.save(r"$bmpOut", format="BMP")
"@
    $python | python -
    if (-not (Test-Path $bmpOut)) {
        throw "Failed to create dialog BMP from $SourceImagePath. Ensure Pillow is installed."
    }
    return $bmpOut
}

function Get-VersionInfo([string]$VersionPath) {
    $resolved = Resolve-ProjectPath $VersionPath
    if (-not $resolved) {
        throw "Version file not found. Provide -VersionFile or create $VersionPath"
    }
    $version = (Get-Content -Path $resolved -TotalCount 1).Trim()
    if (-not $version) {
        throw "Version file is empty: $VersionPath"
    }
    $parts = $version.Split(".")
    while ($parts.Count -lt 4) {
        $parts += "0"
    }
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    $patch = [int]$parts[2]
    $build = [int]$parts[3]
    return @{
        Version = $version
        Major = $major
        Minor = $minor
        Patch = $patch
        Build = $build
    }
}

function New-VersionFile([hashtable]$VersionInfo) {
    $versionFilePath = Join-Path $buildDir "version_info.txt"
    $fileVersion = "{0}.{1}.{2}.{3}" -f $VersionInfo.Major, $VersionInfo.Minor, $VersionInfo.Patch, $VersionInfo.Build
    $versionLines = @(
        "VSVersionInfo(",
        "  ffi=FixedFileInfo(",
        "    filevers=($($VersionInfo.Major), $($VersionInfo.Minor), $($VersionInfo.Patch), $($VersionInfo.Build)),",
        "    prodvers=($($VersionInfo.Major), $($VersionInfo.Minor), $($VersionInfo.Patch), $($VersionInfo.Build)),",
        "    mask=0x3f,",
        "    flags=0x0,",
        "    OS=0x4,",
        "    fileType=0x1,",
        "    subtype=0x0,",
        "    date=(0, 0)",
        "  ),",
        "  kids=[",
        "    StringFileInfo([",
        "      StringTable(",
        "        '040904B0',",
        "        [",
        "          StringStruct('CompanyName', '$CompanyName'),",
        "          StringStruct('FileDescription', 'Data Snipper'),",
        "          StringStruct('FileVersion', '$fileVersion'),",
        "          StringStruct('InternalName', '$AppName'),",
        "          StringStruct('OriginalFilename', '$AppName.exe'),",
        "          StringStruct('ProductName', 'Data Snipper'),",
        "          StringStruct('ProductVersion', '$fileVersion')",
        "        ]",
        "      )",
        "    ]),",
        "    VarFileInfo([VarStruct('Translation', [1033, 1200])])",
        "  ]",
        ")"
    )
    $versionLines | Set-Content -Path $versionFilePath -Encoding ASCII
    return $versionFilePath
}

$versionInfo = Get-VersionInfo $VersionFile
$versionFilePath = New-VersionFile $versionInfo

function New-LicenseRtf([string]$LicensePath) {
    $resolved = Resolve-ProjectPath $LicensePath
    if (-not $resolved) {
        throw "License file not found. Provide -LicensePath or create $LicensePath"
    }
    $licenseText = Get-Content -Path $resolved -Raw
    if (-not $licenseText) {
        throw "License file is empty: $LicensePath"
    }
    $escaped = $licenseText -replace '\\', '\\' -replace '{', '\{' -replace '}', '\}'
    $escaped = $escaped -replace "`r`n", "\par`r`n" -replace "`n", "\par`n"
    $rtf = "{\rtf1\ansi\deff0`r`n" +
        "{\fonttbl{\f0 Segoe UI;}}`r`n" +
        "\viewkind4\uc1\fs18`r`n" +
        $escaped + "`r`n}"
    $rtfPath = Join-Path $buildDir "license.rtf"
    $rtf | Set-Content -Path $rtfPath -Encoding ASCII
    return $rtfPath
}

function Invoke-PyInstallerBuild([string]$BuildName, [switch]$UseOneFile) {
    $pyiArgs = @(
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name", $BuildName,
        "--icon", $iconForExe,
        "--splash", $splashResolved.Path,
        "--runtime-hook", $hookPath,
        "--version-file", $versionFilePath
    )

    if ($UseOneFile) {
        $pyiArgs += "--onefile"
    } else {
        $pyiArgs += "--onedir"
    }

    $addBinaryArgs = @()
    $addBinaryArgs += @("--add-binary", "$tessExe;Tesseract-OCR")
    Get-ChildItem -Path $tessRoot -Filter "*.dll" | ForEach-Object {
        $addBinaryArgs += @("--add-binary", "$($_.FullName);Tesseract-OCR")
    }

    $addDataArgs = @(
        "--add-data", "$tessData;Tesseract-OCR\\tessdata",
        "--add-data", "$($iconResolved.Path);images",
        "--add-data", "$iconRuntimePath;images",
        "--add-data", "$($splashResolved.Path);images"
    )

    $pyiArgs += $addBinaryArgs
    $pyiArgs += $addDataArgs
    $pyiArgs += "snipper.py"

    Write-Host "Running: python -m PyInstaller $($pyiArgs -join ' ')"
    & python -m PyInstaller @pyiArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
}

function Test-WixBin([string]$BinPath) {
    if ([string]::IsNullOrWhiteSpace($BinPath)) {
        return $false
    }
    return (Test-Path (Join-Path $BinPath "candle.exe")) -and
        (Test-Path (Join-Path $BinPath "light.exe")) -and
        (Test-Path (Join-Path $BinPath "heat.exe"))
}

function Resolve-WixToolsetBin([string]$OverridePath) {
    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($OverridePath)) {
        $resolved = Resolve-Path -Path $OverridePath -ErrorAction SilentlyContinue
        if (-not $resolved) {
            throw "WiX Toolset path not found: $OverridePath"
        }
        $resolvedPath = $resolved.Path
        if ((Get-Item $resolvedPath).PSIsContainer) {
            $candidates += $resolvedPath
        } else {
            $candidates += (Split-Path -Parent $resolvedPath)
        }
    }

    if ($env:WIX) {
        $candidates += (Join-Path $env:WIX "bin")
        $candidates += $env:WIX
    }

    $cmd = Get-Command candle.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $candidates += (Split-Path -Parent $cmd.Source)
    }

    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles "WiX Toolset v3.11\\bin"
        $candidates += Join-Path $env:ProgramFiles "WiX Toolset v3.14\\bin"
        $candidates += Join-Path $env:ProgramFiles "WiX Toolset v4\\bin"
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates += Join-Path ${env:ProgramFiles(x86)} "WiX Toolset v3.11\\bin"
        $candidates += Join-Path ${env:ProgramFiles(x86)} "WiX Toolset v3.14\\bin"
        $candidates += Join-Path ${env:ProgramFiles(x86)} "WiX Toolset v4\\bin"
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-WixBin $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function New-WixInstaller([string]$SourceDir, [string]$TargetMsi) {
    $sourceFull = (Resolve-Path $SourceDir).Path
    if (-not (Test-Path $sourceFull)) {
        throw "Source folder not found: $SourceDir"
    }

    $wixBin = Resolve-WixToolsetBin $WixPath
    if (-not $wixBin) {
        throw "WiX Toolset not found. Install WiX Toolset v3.x and pass -WixPath if it is not on PATH."
    }

    $heat = Join-Path $wixBin "heat.exe"
    $candle = Join-Path $wixBin "candle.exe"
    $light = Join-Path $wixBin "light.exe"

    $wxsMain = Join-Path $buildDir "installer.wxs"
    $wxsHarvest = Join-Path $buildDir "harvest.wxs"
    $appId = "{7B7D7037-1AE0-4E10-9F4C-7D7E1D0A6C2D}"
    $appVersion = $versionInfo.Version
    $licenseRtf = New-LicenseRtf "LICENSE"
    $dialogBmp = New-DialogBmp $splashResolved.Path 493 312 160 "wix_dialog.bmp"
    $bannerResolved = Resolve-ProjectPath "images\\snipper-banner.png"
    if (-not $bannerResolved) {
        throw "Banner image not found. Create images\\snipper-banner.png or adjust the build script."
    }
    $bannerBmp = New-BannerBmp $bannerResolved.Path 493 58 160 "wix_banner.bmp"

    $heatArgs = @(
        "dir", $sourceFull,
        "-cg", "AppFiles",
        "-dr", "INSTALLFOLDER",
        "-gg",
        "-scom",
        "-sreg",
        "-srd",
        "-var", "var.SourceDir",
        "-out", $wxsHarvest
    )
    Write-Host "Running: `"$heat`" $($heatArgs -join ' ')"
    & $heat @heatArgs
    if ($LASTEXITCODE -ne 0) {
        throw "WiX heat failed."
    }

    $wxs = @()
    $wxs += "<?xml version=`"1.0`" encoding=`"UTF-8`"?>"
    $wxs += "<Wix xmlns=`"http://schemas.microsoft.com/wix/2006/wi`">"
    $wxs += "  <Product Id=`"*`" Name=`"$AppName`" Manufacturer=`"$CompanyName`" Version=`"$appVersion`" Language=`"1033`" UpgradeCode=`"$appId`">"
    $wxs += "    <Package InstallerVersion=`"500`" Compressed=`"yes`" InstallScope=`"perMachine`" />"
    $wxs += "    <MediaTemplate EmbedCab=`"yes`" />"
    $wxs += "    <Icon Id=`"AppIcon`" SourceFile=`"$iconForExe`" />"
    $wxs += "    <Property Id=`"ARPPRODUCTICON`" Value=`"AppIcon`" />"
    $wxs += "    <WixVariable Id=`"WixUILicenseRtf`" Value=`"$licenseRtf`" />"
    $wxs += "    <WixVariable Id=`"WixUIDialogBmp`" Value=`"$dialogBmp`" />"
    $wxs += "    <WixVariable Id=`"WixUIBannerBmp`" Value=`"$bannerBmp`" />"
    $wxs += "    <Property Id=`"WIXUI_INSTALLDIR`" Value=`"INSTALLFOLDER`" />"
    $wxs += "    <UIRef Id=`"WixUI_Minimal`" />"
    $wxs += "    <UIRef Id=`"WixUI_ErrorProgressText`" />"
    $wxs += "    <Directory Id=`"TARGETDIR`" Name=`"SourceDir`">"
    $wxs += "      <Directory Id=`"ProgramFilesFolder`">"
    $wxs += "        <Directory Id=`"INSTALLFOLDER`" Name=`"$AppName`" />"
    $wxs += "      </Directory>"
    $wxs += "      <Directory Id=`"ProgramMenuFolder`">"
    $wxs += "        <Directory Id=`"AppProgramMenuFolder`" Name=`"$AppName`" />"
    $wxs += "      </Directory>"
    $wxs += "    </Directory>"
    $wxs += "    <DirectoryRef Id=`"AppProgramMenuFolder`">"
    $wxs += "      <Component Id=`"StartMenuShortcut`" Guid=`"*`">"
    $wxs += "        <Shortcut Id=`"StartMenuShortcut`" Name=`"$AppName`" Target=`"[INSTALLFOLDER]$AppName.exe`" WorkingDirectory=`"INSTALLFOLDER`" Icon=`"AppIcon`" />"
    $wxs += "        <RemoveFolder Id=`"CleanProgramMenu`" On=`"uninstall`" />"
    $wxs += "        <RegistryValue Root=`"HKCU`" Key=`"Software\$AppName`" Name=`"Installed`" Type=`"integer`" Value=`"1`" KeyPath=`"yes`" />"
    $wxs += "      </Component>"
    $wxs += "    </DirectoryRef>"
    $wxs += "    <Feature Id=`"MainFeature`" Title=`"$AppName`" Level=`"1`">"
    $wxs += "      <ComponentGroupRef Id=`"AppFiles`" />"
    $wxs += "      <ComponentRef Id=`"StartMenuShortcut`" />"
    $wxs += "    </Feature>"
    $wxs += "  </Product>"
    $wxs += "</Wix>"

    $wxs | Set-Content -Path $wxsMain -Encoding ASCII

    $candleOut = "$buildDir\\"
    $candleArgs = @(
        "-dSourceDir=$sourceFull",
        "-out", $candleOut,
        $wxsMain,
        $wxsHarvest
    )
    Write-Host "Running: `"$candle`" $($candleArgs -join ' ')"
    & $candle @candleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "WiX candle failed."
    }

    $wixObjMain = Join-Path $buildDir "installer.wixobj"
    $wixObjHarvest = Join-Path $buildDir "harvest.wixobj"
    $lightArgs = @(
        "-ext", "WixUIExtension",
        "-sice:ICE60",
        "-out", $TargetMsi,
        $wixObjMain,
        $wixObjHarvest
    )
    Write-Host "Running: `"$light`" $($lightArgs -join ' ')"
    & $light @lightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "WiX light failed."
    }
}

if ($buildSingleFile) {
    $oneFileName = "$AppName-$($versionInfo.Version)-onefile"
    Invoke-PyInstallerBuild -BuildName $oneFileName -UseOneFile
}

if ($buildSfx) {
    $oneDirName = "$AppName"
    Invoke-PyInstallerBuild -BuildName $oneDirName
    $distDir = Join-Path $PSScriptRoot "dist"
    $oneDirOut = Join-Path $distDir $oneDirName
    $msiOut = Join-Path $distDir "$AppName-$($versionInfo.Version)-installer.msi"
    New-WixInstaller -SourceDir $oneDirOut -TargetMsi $msiOut
}
