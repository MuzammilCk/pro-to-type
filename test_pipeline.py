import cv2
from detector import Detector
from reasoner import Reasoner
from alerter import Alerter

det = Detector("models/yolov5s.onnx")
reasoner = Reasoner()
alerter = Alerter(enabled=False)

frame = cv2.imread("samples/test_frame.jpg")
dets = det.detect(frame)
print("detections:", [(d.label, d.confidence) for d in dets])

decision = reasoner.evaluate(dets)
print("decision:", decision["should_act"])
print("threats:", [t.label for t in decision["threats"]])

for d in dets:
    det.draw(frame, d)

label = f"dets={decision['total']} act={decision['should_act']}"
cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.imwrite("samples/output_test.jpg", frame)
print("output saved to samples/output_test.jpg")
