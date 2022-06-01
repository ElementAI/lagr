#!/bin/bash -l
#!/bin/bash -l
export PARENT_DIR=cfq/sparql

python cfq/preprocess_cfq.py --dir ${PARENT_DIR}

export CFQ_SLPIT=(random_split mcd1 mcd2 mcd3)

for SPLIT in ${CFQ_SLPIT[@]}
do
        export DATADIR=${PARENT_DIR}/${SPLIT}/*

        for f in $DATADIR
        do
                FILE_BASENAME="${f##*/}"

                if [ "${FILE_BASENAME: -4}" == ".tsv" ]
                then
                        echo 'Building CFQ graphs for '${SPLIT} '- '$FILE_BASENAME
                        python cfq/graph_parser.py --data $FILE_BASENAME --split ${SPLIT} --dir ${PARENT_DIR}
                fi
        done
        echo 'Done building CFQ graphs.'
done
