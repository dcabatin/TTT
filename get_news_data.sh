mkdir data
wget http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/en-fr.txt.zip -O tmp-data.zip
unzip tmp-data.zip -d data/news
rm tmp-data.zip
