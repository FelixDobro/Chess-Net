scp -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" "C:\Users\Felix Dobrovich\Desktop\Data\initial.zip" Ubuntu@remote-server:Chess-Net/data/
ssh -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" Ubuntu@remote-server 'mkdir -p Chess-Net/data/training_chunk_data && unzip Chess-Net/data/initial.zip -d Chess-Net/data/training_chunk_data'


scp -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" "C:\Users\Felix Dobrovich\Desktop\Data\rest.zip" Ubuntu@remote-server:Chess-Net/data/
ssh -i "C:\Users\Felix Dobrovich\.ssh\a100-key.pem" Ubuntu@remote-server 'mkdir -p Chess-Nesdt/data/training_chunk_data && unzip Chess-Net/data/rest.zip -d Chess-Net/data/training_chunk_data'
