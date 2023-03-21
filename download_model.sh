mkdir model_zoo
filename='team20_megnr_v2.pth'
fileid='1-SZHKS3zLecfBEckWlLX7WbUB6kAeHzr'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ./model_zoo/${filename} && rm -rf /tmp/cookies.txt


