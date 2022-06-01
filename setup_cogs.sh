#!/bin/bash -l

# download dataset
echo 'Setting up dataset under ./data.'
TARGET_DIR='cogs/data'
DATADIR=${TARGET_DIR}'/lambda'
mkdir -p $DATADIR

git clone https://github.com/najoungkim/COGS $DATADIR/original
rm -rf ${DATADIR}/original/data/train_100.tsv

# create gen dev set
echo 'Building gen dev set for tuning...'
export GEN_DEV=${DATADIR}'/original/data/gen_dev.tsv'
<${DATADIR}'/original/data/gen.tsv' sort -R | head -n 1000 > ${GEN_DEV} 
echo $(wc -l ${GEN_DEV})

# removes rows that contain primitives not in the train set
echo 'Removing rows that contain primitives not in the train set...'
for f in $(ls $DATADIR/original/data/*)
do
	echo 'Processing' ${f}
	NEW_FILE=${f##*/}
	grep -i -v -E "gardner|monastery" $f > $DATADIR/$NEW_FILE
done

# remove original files
rm -rf ${DATADIR}/original

echo 'Done filtering.'

# parse data as COGS graphs
echo 'Parsing COGS logical forms as graphs and building vocabularies...'
python src/utils/parser.py --data_dir ${TARGET_DIR}
echo 'Done building COGS graphs.'

# splits src and tgt sequences
python src/utils/preprocess.py --parent_path $TARGET_DIR
