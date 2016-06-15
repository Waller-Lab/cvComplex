PROJECT = libcvComplex.a
OBJECTS = cvComplex.o

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_legacy

# Extensions to clean
CLEANEXTS = o a

INCLUDES = -I/usr/local/include/ -I/usr/local/opt/opencv3/include
# define the C compiler to use
CC = clang++

# define any compile-time flags
CFLAGS= -std=c++14 -ggdb -Wall -pedantic

LFLAGS= -L/usr/local/opt/opencv3/lib

all: $(PROJECT)

.cpp.o:
	$(CC) -c $(CFLAGS) $(INCLUDES) $< $(LFLAGS) $(LIBS)

$(PROJECT): $(OBJECTS)
	ar ru $@ $^

.PHONY: clean install
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done

install:
	cp *.a /usr/local/lib/
	cp *.h /usr/local/include/
