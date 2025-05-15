# AMD - Proof Of Concept
VICIDIAL and AMD using wav2vec-vm-finetune

````
sudo zypper install python310 python310-pip python310-virtualenv
cd /var/lib/asterisk/agi-bin
python3.10 -m venv venv310
source venv310/bin/activate
pip install torch>=2.5.1 torchaudio transformers==4.48.2 soundfile


https://github.com/user-attachments/assets/480c1c37-5457-45d6-950c-2bd5fc3a4a99




exten => 7002,1,NoOp(Test AI AMD)
 ; Answer the call and run your EAGI script
 same  => n,Answer()
 same  => n,EAGI(amd3.py)
 ; Show what we got
 same  => n,NoOp(RESULT: ${AMDSTATUS} / ${AMDCAUSE})

 ; If it’s a MACHINE (voicemail), jump to the vm context
 same  => n,GotoIf($["${AMDSTATUS}"="MACHINE"]?vm,1)
 ; If there was an error, drop the call
 same  => n,GotoIf($["${AMDSTATUS}"="AIERR"]?fail,1)
 ; Otherwise (HUMAN) go on to your normal flow at 8368
 same  => n,Goto(8368,1)

 ; ——— Voicemail / MACHINE path ———
exten => vm,1,NoOp(Voicemail/MACHINE detected – playing audio)
 same  => n,Playback(vm-detected)       ; replace "vm-detected" with your file
 same  => n,Hangup()

 ; ——— Error path ———
exten => fail,1,NoOp(AIAMD error, hanging up)
 same  => n,Hangup()

 ; ——— Your normal human flow at 8368 ———
;  (keep whatever you already have in context default,exten=8368)
