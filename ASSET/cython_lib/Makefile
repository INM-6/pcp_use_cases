#
# Makefile to only rebuild the extension if *.pyx changes
#
CXX = g++
#CXX = icpc
CYTHON = cython
PYTHON = python

#VECTORIZE_OFF = 1


PYTHON_INC = $(shell $(PYTHON) -c 'import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.get_python_inc())')
PYTHON_INC += -I$(shell $(PYTHON) -c 'import numpy; print numpy.get_include()')

# python -c "import numpy; print numpy.get_include()"

ifdef PYTHON_INC
    CPPFLAGS += -I$(PYTHON_INC)
endif

ifdef BOOST_INC
    CPPFLAGS += -I$(BOOST_INC)
endif

ifeq ($(CXX),g++)
    OPENMPFLAGS = -fopenmp
    OPTFLAGS = -O3 -Wall -march=native -fno-strict-aliasing -fno-associative-math
	ifdef VECTORIZE_OFF
		OPTFLAGS += -fno-tree-vectorize
	endif
    #CPPFLAGS += -I/gpfs/homeb/pcp0/pcp0016/.local/lib/python2.7/site-packages/numpy/core/include/
endif

ifeq ($(CXX),icpc)
    CXXFLAGS = -m64
	# For knl: -xmic-avx512 -xCOMMON-AVX512
    OPTFLAGS =  -qopt-report=5 -O3 -no-vec
	#CPPFLAGS += -I/gpfs/homeb/pcp0/pcp0016/.local/lib/python2.7/site-packages/numpy/core/include/
	ifdef VECTORIZE_OFF
		OPTFLAGS += -no-vec
	endif

    OPENMPFLAGS = -openmp
endif

CXXFLAGS += -shared -DPIC -fPIC $(OPTFLAGS) $(OPENMPFLAGS)

# Put all your target modules in here
#
MODULES = _routines.pyx

all: $(MODULES:%.pyx=%.so) $(MODULES:%.pyx=%.html)

.SECONDARY: $(MODULES:%.pyx=%.cpp)

%.so: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.cpp %.html: %.pyx
	$(CYTHON) --cplus --annotate $<

clean:
	@rm -f *.c *.cpp *.html *.o *.so *~ *.pyc core*

.PHONY: clean
