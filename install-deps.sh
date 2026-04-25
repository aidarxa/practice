#!/bin/bash
git clone https://github.com/hyrise/sql-parser.git
cd sql-parser
make
cd ..
mkdir -p deps/lib && mkdir -p deps/include 
mkdir -p deps/include/parser && mkdir -p deps/include/sql && mkdir -p deps/include/util
mv sql-parser/libsqlparser.so deps/lib/libsqlparser.so
mv sql-parser/src/*.h deps/include/
mv sql-parser/src/parser/*.h deps/include/parser/
mv sql-parser/src/sql/*.h deps/include/sql/
mv sql-parser/src/util/*.h deps/include/util/