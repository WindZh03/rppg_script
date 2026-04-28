import cv2

video_path = "/share2/data/zhaoqiqi/dataset/rppg/CASME_2/rawvideo/s15/15_0101disgustingteeth.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps if fps > 0 else None

cap.release()

print("fps:", fps)
print("frame_count:", frame_count)
print("duration:", duration)
