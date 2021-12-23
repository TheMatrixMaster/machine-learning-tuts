import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.rectangle(frame, (15, 25), (200, 150), (255, 0, 0), 1)
	cv2.circle(frame, (100, 63), 55, (0, 0, 255), -1)

	pts = np.array(([10, 5], [20, 30], [70, 20], [50, 10]), np.int32)

	cv2.polylines(frame, [pts], True, (255, 255, 255), 5)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, 'OpenCV!', (0, 130), font, 3, (200, 255, 255), 5)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()