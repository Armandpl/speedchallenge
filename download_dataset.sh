mkdir data
curl -LJO https://github.com/commaai/speedchallenge/blob/master/data/train.mp4?raw=true
curl -LJO https://github.com/commaai/speedchallenge/blob/master/data/train.txt?raw=true
curl -LJO https://github.com/commaai/speedchallenge/blob/master/data/test.mp4?raw=true
mv train.mp4?raw=true data/train.mp4
mv test.mp4?raw=true data/test.mp4
mv train.txt?raw=true data/train.txt