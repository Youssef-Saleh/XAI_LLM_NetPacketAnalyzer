@echo off
REM Build thesis PDF using pdflatex + biber
pdflatex thesis_main.tex
biber thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex
