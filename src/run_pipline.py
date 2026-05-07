from .pipline.video_pipeline import VideoPipeline

if __name__ == "__main__":
    source = "videos/video1.mp4"

    pipeline = VideoPipeline(source)
    pipeline.process()  