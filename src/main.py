from .pipeline import VideoPipeline
import cv2

if __name__ == "__main__":
    video_path = "videos/video2.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video açılamadı.")

    pipeline = VideoPipeline(video_path)
    pipeline.process()