PROJECT = libcvComplex.a
OBJECTS = cvComplex.o

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib

# Extensions to clean
CLEANEXTS   = o a 

# define the C compiler to use
CC=g++

# define any compile-time flags
CFLAGS= -std=c++14 -ggdb -Wall -pedantic

all: $(PROJECT)

.cpp.o:
	$(CC) -c $(CFLAGS) $< $(LFLAGS) $(LIBS)

$(PROJECT): $(OBJECTS)
	ar ru $@ $^
	
.PHONY: clean install
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done
	
install:
	cp *.a /usr/lib/
