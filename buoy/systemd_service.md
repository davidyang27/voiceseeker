1. 建立 systemd 服務檔：

sudo nano  /etc/systemd/system/multi-user.target.wants/buoy_acoustic_system.service 

2. 填入內容：

$ cat /etc/systemd/system/multi-user.target.wants/buoy_acoustic_system.service 
[Unit]
Description=ST6100 Transmit Service
After=sound.target

[Service]
ExecStart=/usr/bin/python3 /home/david/Desktop/voiceseeker/buoy_acoustic_system.py
WorkingDirectory=/home/david/Desktop/skybee
StandardOutput=file:/home/david/Desktop/buoy_acoustic_system_log/output.log
StandardError=file:/home/david/Desktop/buoy_acoustic_system_log/output.err
Restart=always
User=david
Environment=PYTHONUNBUFFERED=1
RestartSec=10
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target

```
cd ~/Desktop/voiceseeker/buoy/ && \
sudo cp buoy_acoustic_system.service /lib/systemd/system/ %% \
cd /etc/systemd/system/multi-user.target.wants/ && \
sudo ln -sf /lib/systemd/system/buoy_acoustic_system.service buoy_acoustic_system.service 
```

3. 啟用並啟動服務： (after editing the buoy_acoustic_system.service)

sudo systemctl daemon-reload && \
sudo systemctl enable buoy_acoustic_system.service && \
sudo systemctl start buoy_acoustic_system.service

4. 檢查狀態：

sudo systemctl status buoy_acoustic_system.service







5. 讀取程式輸出

假設你的 service 名稱叫 buoy_acoustic_system.service，可以用：

sudo journalctl -u buoy_acoustic_system.service -f



6. 停止程式
sudo systemctl stop buoy_acoustic_system.service


7. 禁用開機自啟

如果你不想每次開機自動啟動：

sudo systemctl disable buoy_acoustic_system.service


8. 重新啟動程式
sudo systemctl restart buoy_acoustic_system.service






Satellite (ST6100)
1. connect to ST6100

sudo apt install picocom
picocom -b 9600 /dev/ttyUSB0

2. set ST6100 default mode at AT command

shell> stop
#cui sp
sp# config system defInterface at wr


