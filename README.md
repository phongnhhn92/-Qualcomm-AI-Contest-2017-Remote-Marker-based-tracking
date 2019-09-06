# Remote-Marker-based-Tracking-using-MobileNet-SSD
This is the demo app for my project: Remote Marker based tracking for accurate drone landing for the 2017 Qualcomm AI Developer Contest https://developforai.com/

Demonstration video: https://www.youtube.com/watch?v=Lnm4uiw3tv4

Please follow these steps in order to run my app.

Installation step:
1. Copy Marker folder into DCIM\Camera\ manually
2. Install marker-detection-release.apk on the device
3. At the first time the app open, please allow all the required permissions.

Demo:
1. Click "Load Model" to load the pretrained MobileNet-SSD Marker detection
2. Click "Detection" button to begin marker detection in all images of "Marker" folder above.
These test images are not included in my training data. If you want to have further test then you can download additional data from this link: https://www.dropbox.com/s/vzyl12jrc3dyflk/Marker_full.zip?dl=0 (please change the folder name to "Marker" if you want to use it)
3. The result images are saved at the folder DetectedResult_test as png format.
