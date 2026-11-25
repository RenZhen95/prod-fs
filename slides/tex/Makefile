# === === === === === === === === === === === 
# @Author : Liaw
# @Date   : 08.07.2025
#
# Makefile to compile a LaTeX project using
# your own customized Docker image based on
# texlive's official Docker image
# 
# === === === === === === === === === === ===

# Name of the the "main" file to compile
NAME=Slides

DOCKER_LATEXCOMPILER=liawlatex

all:
	docker run --rm -v "$(shell pwd):/project:Z" -w /project $(DOCKER_LATEXCOMPILER) latexmk -pdf $(NAME).tex

clean:
	rm -f *.toc
	rm -f $(NAME).dvi
	rm -f *.log
	rm -f *.aux
	rm -f $(NAME).ps
	rm -f $(NAME).pdf
	rm -f *~
	rm -f *.fls
	rm -f *.fdb_latexmk
