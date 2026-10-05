@echo off
rem nsf2mid GUI. Drop an .nsf file on this file to open it.
start "" pythonw "%~dp0src\nsf2mid_gui.py" %*
