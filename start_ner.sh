cd /root/ner
git pull
docker stop ner
docker rm ner
docker build -t ner .
docker run -itd -p 8077:8077 --name ner ner /bin/bash
