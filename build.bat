@echo off
echo Building the app...

REM Clean old builds
rmdir /s /q build
rmdir /s /q dist
del app.spec

REM Run PyInstaller with the necessary options
pyinstaller --noconsole --noconfirm --onedir --clean --hidden-import numpy._core._multiarray_tests --add-data "best.pt;." app.py

echo Build complete! Find your app in the "dist" folder.
pause
