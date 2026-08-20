.PHONY: all clean
all:
	pdflatex thesis_main.tex
	biber thesis_main
	pdflatex thesis_main.tex
	pdflatex thesis_main.tex
clean:
	rm -f thesis_main.aux thesis_main.log thesis_main.bbl thesis_main.bcf
	rm -f thesis_main.blg thesis_main.toc thesis_main.out
	rm -f thesis_main.lof thesis_main.lot thesis_main.loa
	rm -f thesis_main.run.xml thesis_main.synctex.gz
