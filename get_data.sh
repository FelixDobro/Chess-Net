scp -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" "C:\Users\Felix Dobrovich\Desktop\Data\initial.zip"ssh ubuntu@129.213.94.58:Chess-Net/data/
ssh -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" ssh ubuntu@129.213.94.58 'mkdir -p Chess-Net/data/training_chunk_data && unzip Chess-Net/data/initial.zip -d Chess-Net/data/training_chunk_data'


scp -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" "C:\Users\Felix Dobrovich\Desktop\Data\rest.zip"ssh ubuntu@129.213.94.58:Chess-Net/data/
ssh -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" ssh ubuntu@129.213.94.58 'mkdir -p Chess-Nesdt/data/training_chunk_data && unzip Chess-Net/data/rest.zip -d Chess-Net/data/training_chunk_data'
