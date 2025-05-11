#!/bin/bash

EXEC=./build/thread
INPUT=Input
OUTPUT=Output
ZOOM=6
NORM=2
for image in $*; do
    outfile=`basename ${image} | sed 's/.png//'`
    echo "---- processing ${outfile} -----"
    for nails in 50 100 150; do
	for thick in 0.2 0.1 0.05 0.025; do
	    dthick=`echo $thick | sed 's/\./_/'`
	    outname="${outfile}-n${nails}-t${dthick}-lp${NORM}"
	    echo "    >>>> computing ${outname} <<<<"
	    ${EXEC} -i ${image} -z ${ZOOM} -t ${thick} -n ${nails} -p ${NORM} -c 2.0 -o ${OUTPUT}/${outname}
	done
    done
done
