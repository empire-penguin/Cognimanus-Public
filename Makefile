all: main.py 
	python3 main.py

collect_data: legacy/emg_task0.py
	python legacy/emg_task0.py

live_data: legacy/live_lsl_anim.py
	python legacy/live_lsl_anim.py

start_lsl_recorder: 
	open /usr/local/opt/labrecorder/LabRecorder/LabRecorder.app

process_data: 
	python legacy/process_emg.py

test:
	python test.py

model:
	python3 hand_pred_model.py