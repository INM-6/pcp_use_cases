#
# Makefile to only rebuild the extension if *.pyx or *.hpp changes
#
CXX = g++
CYTHON = cython
PYTHON = python

PYTHON_INC = $(shell $(PYTHON) -c 'import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.get_python_inc())')

ifdef PYTHON_INC
    CPPFLAGS += -I$(PYTHON_INC)
endif

ifdef BOOST_INC
    CPPFLAGS += -I$(BOOST_INC)
endif

ifeq ($(CXX),g++)
    OPENMPFLAGS = -fopenmp
    OPTFLAGS = -O3 -Wall -march=native -fno-strict-aliasing -fno-associative-math
	CPPFLAGS += $(shell python2-config --includes) $(shell python -c "import numpy; print '-I' + numpy.get_include()")
	#http://stackoverflow.com/questions/9019580/makefile-for-c-program-that-uses-numpy-extensions
endif

ifeq ($(CXX),icpc)
    CXXFLAGS = -m64
    OPTFLAGS = -O3 -fp-model strict -fomit-frame-pointer -xhost
    OPENMPFLAGS = -openmp
endif

CXXFLAGS += -shared -DPIC -fPIC $(OPTFLAGS) $(OPENMPFLAGS)

# Put all your target modules in here
#
MODULES = _routines.pyx _routines.hpp

all: $(MODULES:%.pyx=%.so) $(MODULES:%.pyx=%.html)

.SECONDARY: $(MODULES:%.pyx=%.cpp)

%.so: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.cpp %.html: %.pyx
	$(CYTHON) --cplus --annotate $<

clean:
	@rm -f *.c *.cpp *.html *.o *.so *~ *.pyc core*

.PHONY: clean
