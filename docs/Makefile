# Makefile for Sphinx documentation
# for the transmon_fluxonium_sim package
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean genapi htmlfast

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto
	rm -rf api/generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# generate api docs
genapi:
	sphinx-apidoc  -o  _auto  -d 10 ../transmon_fluxonium_sim
	mkdir -p api/generated/
	cp _auto/* api/generated/

# faster build by skipping execution of all notebooks
htmlfast: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute=never

html: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
