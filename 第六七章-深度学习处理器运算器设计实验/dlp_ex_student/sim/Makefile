# Module and test sources
MODULES = inner_product_4x16 matrix_vector_mult_4x4x16 matrix_mult_4x4x4x16
TEST_FILES = tb_inner_product.cpp tb_matrix_vector.cpp tb_matrix_mult.cpp

# Paths (robust when running from repo root or sim/)
ROOT_DIR ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
SRC_DIR ?= $(ROOT_DIR)/src
SIM_DIR ?= $(ROOT_DIR)/sim
OBJ_DIR ?= $(SIM_DIR)/obj_dir

# Verilator config
VERILATOR = verilator
VERILATOR_FLAGS = --cc --exe --build -Wall --trace -I$(SRC_DIR)
CXXFLAGS = -std=c++14

# Default target
all: $(MODULES)

# Inner product
inner_product_4x16:
	@echo "=== build $@ ==="
	mkdir -p $(OBJ_DIR)/$@
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		$(SIM_DIR)/tb_inner_product.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# Matrix x vector
matrix_vector_mult_4x4x16:
	@echo "=== build $@ ==="
	mkdir -p $(OBJ_DIR)/$@
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		$(SRC_DIR)/inner_product_4x16.v \
		$(SIM_DIR)/tb_matrix_vector.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# Matrix x matrix
matrix_mult_4x4x4x16:
	@echo "=== build $@ ==="
	mkdir -p $(OBJ_DIR)/$@
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		$(SRC_DIR)/matrix_vector_mult_4x4x16.v \
		$(SRC_DIR)/inner_product_4x16.v \
		$(SIM_DIR)/tb_matrix_mult.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# Cleanup
clean:
	@echo "=== clean ==="
	rm -rf $(OBJ_DIR)
	rm -f $(SIM_DIR)/*.vcd

# Phony targets
.PHONY: all clean $(MODULES)
