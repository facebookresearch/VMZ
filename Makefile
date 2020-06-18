.PHONY: ops
ops:
	mkdir -p build && cd build && cmake .. && make -j$(shell nproc)

.PHONY: clean
clean:
	rm -rf build
