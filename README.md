# AMD - Proof Of Concept
VICIDIAL and AMD using wav2vec-vm-finetune

````
sudo zypper install python310 python310-pip python310-virtualenv
cd /var/lib/asterisk/agi-bin
python3.10 -m venv venv310
source venv310/bin/activate
pip install torch>=2.5.1 torchaudio transformers==4.48.2 soundfile


https://github.com/user-attachments/assets/480c1c37-5457-45d6-950c-2bd5fc3a4a99




; VICIDIAL_auto_dialer transfer script AMD with Load Balanced:
exten => 8369,1,AGI(agi://127.0.0.1:4577/call_log)
exten => 8369,n,Playback(sip-silence)
exten => 8369,n,EAGI(amd3.py)  ; <- AI detection instead of AMD()
exten => 8369,n,NoOp(RESULT: ${AMDSTATUS} / ${AMDCAUSE})
exten => 8369,n,GotoIf($["${AMDSTATUS}" = "MACHINE"]?vm,1)
exten => 8369,n,GotoIf($["${AMDSTATUS}" = "AIERR"]?fail,1)


;exten => 8369,n,AMD(2000,2000,1000,5000,120,50,4,256) 
;exten => 8369,n,AGI(VD_amd.agi,${EXTEN})
exten => 8369,n,AGI(agi-VDAD_ALL_outbound.agi,NORMAL-----LB-----${CONNECTEDLINE(name)})
exten => 8369,n,Hangup()
