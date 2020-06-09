
# make "all" the target.
all:


# Disable linking math libs because not needed here. Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =

include ../kaldi.mk


ifeq ($(KALDI_FLAVOR),static)
	$(error You cannot build the pybind directory unless the build was dynamic; reconfigure with --shared option.)
endif

PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes)

CXXFLAGS += $(PYBIND_INCLUDES)

TARGET_PACKAGENAME = kaldi_pybind
PYBIND_EXTENSION = $(shell python3-config --extension-suffix)
LIBFILE_EXTENSION = $(shell basename $(PYBIND_EXTENSION) .so)
LIBNAME = $(TARGET_PACKAGENAME)$(LIBFILE_EXTENSION)

# pybind11 is heavily templated and generates code that is bloated before optimization.
# -flto is link time optimization which apparently is important.
ifndef CI_TARGETS
ifneq ($(shell uname), Darwin)
	LTOFLAG = -flto
endif
endif

CXXFLAGS += -O3 $(LTOFLAG) -I.

ifneq ($(shell uname), Darwin) # llvm does not have feature like ld.gold
	LDFLAGS += -fuse-ld=gold
endif

ifeq ($(shell uname),Darwin)
	LDFLAGS += -undefined dynamic_lookup
	EXTRA_LDLIBS += ../../tools/openfst/lib/libfstscript.dylib
else
	EXTRA_LDLIBS += ../../tools/openfst/lib/libfstscript.so
endif

OBJFILES = chain/chain_den_graph_pybind.o chain/chain_pybind.o chain/chain_supervision_pybind.o \
           chain/chain_training_pybind.o ctc/ctc_pybind.o cudamatrix/cu_device_pybind.o \
           cudamatrix/cu_matrix_pybind.o cudamatrix/cu_vector_pybind.o cudamatrix/cudamatrix_pybind.o \
           decoder/decodable_matrix_pybind.o decoder/decoder_pybind.o decoder/decoder_wrappers_pybind.o \
           decoder/lattice_faster_decoder_pybind.o dlpack/dlpack_deleter.o dlpack/dlpack_pybind.o \
           dlpack/dlpack_submatrix.o dlpack/dlpack_subvector.o feat/feat_pybind.o feat/feature_pybind.o \
           feat/online_feature_pybind.o feat/wave_reader_pybind.o fst/arc_pybind.o fst/compile_pybind.o \
           fst/fst_pybind.o fst/symbol_table_pybind.o fst/vector_fst_pybind.o fst/weight_pybind.o \
           fstext/fstext_pybind.o fstext/kaldi_fst_io_pybind.o fstext/lattice_weight_pybind.o \
           hmm/hmm_pybind.o hmm/hmm_topology_pybind.o hmm/transition_model_pybind.o \
           itf/context_dep_itf_pybind.o itf/decodable_itf_pybind.o itf/itf_pybind.o \
           itf/options_itf_pybind.o kaldi_pybind.o lat/determinize_lattice_pruned_pybind.o \
           lat/kaldi_lattice_pybind.o lat/lat_pybind.o matrix/compressed_matrix_pybind.o \
           matrix/kaldi_matrix_pybind.o matrix/kaldi_vector_pybind.o matrix/matrix_common_pybind.o \
           matrix/matrix_pybind.o matrix/sparse_matrix_pybind.o nnet3/nnet3_pybind.o \
           nnet3/nnet_chain_example_pybind.o nnet3/nnet_common_pybind.o \
           nnet3/nnet_component_itf_pybind.o nnet3/nnet_convolutional_component_pybind.o \
           nnet3/nnet_example_pybind.o nnet3/nnet_nnet_pybind.o nnet3/nnet_normalize_component_pybind.o \
           nnet3/nnet_simple_component_pybind.o tests/test_dlpack_subvector.o util/kaldi_holder_pybind.o \
           util/kaldi_io_pybind.o util/parse_options_pybind.o util/table_types_pybind.o util/util_pybind.o

# libs should be sorted in alphabetic order
ADDLIBS = ../base/kaldi-base.a ../chain/kaldi-chain.a ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
		  ../feat/kaldi-feat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../lat/kaldi-lat.a \
		  ../matrix/kaldi-matrix.a ../nnet3/kaldi-nnet3.a ../util/kaldi-util.a

ifdef MKL_LD_PRELOAD
LD_PRELOAD += $(MKL_LD_PRELOAD)
export LD_PRELOAD
endif

.PHONY: all clean test

# include the "warp-ctc" make process under subdir: ctc, and use it through pybing11
include ctc/ctc.mk

TARGET_NAME=$(subst libkaldi,kaldi,$(LIBFILE))
TARGET_LIB=$(TARGET_NAME:.dylib=.so) # if "darwin"

all: $(LIBFILE)
	$(shell ln -sf $(LIBFILE) $(TARGET_LIB))

# valgrind-python.supp is from http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp
# since we do not compile Python from source, we follow the comment in valgrind-python.supp
# to uncomment suppressions for PyObject_Free and PyObject_Realloc.
valgrind:
	valgrind --tool=memcheck --suppressions=./valgrind-python.supp
	python3 -E -tt ./tests/test_matrix.py

include ../makefiles/default_rules.mk

# directories should be sorted in alphabetic order
TEST_DIRS = chain cudamatrix decoder dlpack feat fst fstext hmm lat matrix nnet3 tests util

test:
	for d in $(TEST_DIRS); do make -C $$d test || exit 1; done
	python3 -c 'import kaldi_pybind'  # this line is a test.
