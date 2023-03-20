mkdir model_zoo
filename='team20_megnr.pth'
fileid='18R5k6g_bpsRu8kXB2eHcX_4mhfZzlpKn'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ./model_zoo/${filename} && rm -rf /tmp/cookies.txt


