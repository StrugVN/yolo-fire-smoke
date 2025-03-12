# Using Ultralytics YOLO v11 (assuming similar API to previous versions)
from ultralytics import YOLO

model = YOLO(r'C:\Users\Lenovo\Desktop\work\ctu\TGMT\App\py\best.pt')
print(model.args)  # May contain training arguments