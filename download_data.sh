# NOTE: TO DOWNLOAD THE WMT17 QE AND WMT18 APE TRAINING DATA, PLEASE MANUALLY DOWNLOAD THEM FROM
# https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2613
# https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1974
# AND THEN PLACE THEM INTO THE DIRECTORY WHERE THIS SCRIPT IS PLACED

set -ex

mkdir -p tmp

cd tmp

# unzip manually downloaded data
tar zxvf ../task1_en-de_training-dev.tar.gz
tar zxvf ../en_de_NMT_train_dev.tgz

# download training data
wget http://ufallab.ms.mff.cuni.cz/~popel/indomain_training.zip
unzip indomain_training.zip

wget https://object.pouta.csc.fi/OPUS-OpenOffice/v2/moses/de-en.txt.zip -O OOv2.de-en.txt.zip
wget https://object.pouta.csc.fi/OPUS-OpenOffice/v3/moses/de-en_GB.txt.zip -O OOv3.de-en_GB.txt.zip
wget https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/de-en.txt.zip -O KDE4.de-en.txt.zip
wget https://object.pouta.csc.fi/OPUS-KDEdoc/v1/moses/de-en_GB.txt.zip -O KDEdoc.de-en.txt.zip

unzip -o OOv2.de-en.txt.zip
unzip -o OOv3.de-en_GB.txt.zip
unzip -o KDE4.de-en.txt.zip
unzip -o KDEdoc.de-en.txt.zip

mkdir -p en-de.train
cat sentence_level/train.src train.tok.src.en indomain_training/indomain.de-en.de OpenOffice.de-en.en OpenOffice.de-en_GB.en_GB KDE4.de-en.en KDEdoc.de-en_GB.en_GB > en-de.train/train.en
cat sentence_level/train.pe train.tok.pe indomain_training/indomain.de-en.en OpenOffice.de-en.de OpenOffice.de-en_GB.de KDE4.de-en.de KDEdoc.de-en_GB.de > en-de.train/train.de

wget https://object.pouta.csc.fi/OPUS-ada83/v1/moses/en-ru.txt.zip -O ada83.en-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en_AU-ru.txt.zip -O GNOME.en_AU-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-ru.txt.zip -O KDE4.en-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-KDEdoc/v1/moses/en_GB-ru.txt.zip -O KDEdoc.en_GB-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-OpenOffice/v3/moses/en_GB-ru.txt.zip -O OpenOffice.en_GB-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-PHP/v1/moses/en-ru.txt.zip -O PHP.en-ru.txt.zip
wget https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-ru.txt.zip -O Ubuntu.en-ru.txt.zip

unzip -o ada83.en-ru.txt.zip
unzip -o GNOME.en_AU-ru.txt.zip
unzip -o KDE4.en-ru.txt.zip
unzip -o KDEdoc.en_GB-ru.txt.zip
unzip -o OpenOffice.en_GB-ru.txt.zip
unzip -o PHP.en-ru.txt.zip
unzip -o Ubuntu.en-ru.txt.zip

mkdir -p en-ru.train
cat ada83.en-ru.en GNOME.en_AU-ru.en_AU KDE4.en-ru.en KDEdoc.en_GB-ru.en_GB OpenOffice.en_GB-ru.en_GB PHP.en-ru.en Ubuntu.en-ru.en > en-ru.train/train.en
cat ada83.en-ru.ru GNOME.en_AU-ru.ru KDE4.en-ru.ru KDEdoc.en_GB-ru.ru OpenOffice.en_GB-ru.ru PHP.en-ru.ru Ubuntu.en-ru.ru > en-ru.train/train.ru

wget https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-lv.txt.zip
unzip -o en-lv.txt.zip

mkdir -p en-lv.train
cp EMEA.en-lv.en en-lv.train/train.en
cp EMEA.en-lv.lv en-lv.train/train.lv

# download dev & test data

for l in en-de en-ru;
do
    mkdir -p $l
    for d in traindev test;
    do
        wget https://deep-spin.github.io/docs/data/wmt2019_qe/task1_${l}_${d}.tar.gz
        tar zxvf task1_${l}_${d}.tar.gz -C $l/
    done
done

wget https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/gold_labels_test.tar.gz
tar zxvf gold_labels_test.tar.gz

for t in smt nmt;
do
    for d in dev test;
    do
        mkdir -p en-lv/$d.$t
        cp gold_labels_test/word-level/en_lv.$t/$d.* en-lv/$d.$t/
        cp gold_labels_test/sentence-level/en_lv/$d.$t.hter en-lv/$d.$t/$d.hter
    done
done

cd ..

for l in en-de en-ru;
do
    mkdir -p $l
    cp -r tmp/$l.train $l/train
    for d in dev test;
    do
        cp -r tmp/$l/$d $l/$d
    done
done

mkdir -p en-lv
cp -r tmp/en-lv.train en-lv/train
for d in dev.smt test.smt dev.nmt test.nmt;
do
    cp -r tmp/en-lv/$d en-lv/$d
done

rm -r tmp