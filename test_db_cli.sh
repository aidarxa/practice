#!/bin/bash
cd build
for q in q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43; do
  if [ -f ../test/queries/sql/${q}.sql ]; then
    echo "Running ${q}..."
    cat ../test/queries/sql/${q}.sql | ./db_cli > ${q}_out.txt 2>&1
    echo "Done ${q}."
  fi
done
