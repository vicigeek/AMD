# AMD
VICIDIAL and AMD using wav2vec-vm-finetune

````
sudo zypper install python310 python310-pip python310-virtualenv
cd /var/lib/asterisk/agi-bin
python3.10 -m venv venv310
source venv310/bin/activate
pip install torch>=2.5.1 torchaudio transformers==4.48.2 soundfile
