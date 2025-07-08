dockerimage ?= facebookresearch/vmz
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker run -it --rm --ipc=host -v $(srcdir):/usr/src/app -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

r: run


gpu:
	@docker run --runtime=nvidia -it --rm --ipc=host -v $(srcdir):/usr/src/app -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

g: gpu


.PHONY: install i run r update u gpu g
