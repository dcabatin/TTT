mkdir data
wget http://cs.brown.edu/courses/cs1470/projects/public/hw4-seq2seq/stencil_and_data.zip -O tmp-data.zip
unzip tmp-data.zip -d hansard-temp
mkdir data/hansard
mv hansard-temp/stencil_and_data/data/* data/hansard
rm -rf hansard-temp
rm tmp-data.zip
