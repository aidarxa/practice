#!/bin/bash
set -e

# sql-parser
git clone https://github.com/hyrise/sql-parser.git
cd sql-parser
make
cd ..

# linenoise
git clone https://github.com/antirez/linenoise.git
cd linenoise
gcc -O3 -fPIC -c linenoise.c -o linenoise.o
ar rcs liblinenoise.a linenoise.o
cd ..

mkdir -p deps/lib && mkdir -p deps/include 
mkdir -p deps/include/parser && mkdir -p deps/include/sql && mkdir -p deps/include/util

# sqlparser
mv sql-parser/libsqlparser.so deps/lib/libsqlparser.so
mv sql-parser/src/*.h deps/include/
mv sql-parser/src/parser/*.h deps/include/parser/
mv sql-parser/src/sql/*.h deps/include/sql/
mv sql-parser/src/util/*.h deps/include/util/

# linenoise
mv linenoise/liblinenoise.a deps/lib/liblinenoise.a
mv linenoise/linenoise.h deps/include/linenoise.h

rm -rf sql-parser linenoise