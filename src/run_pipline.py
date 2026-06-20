from .pipline.video_pipeline import VideoPipeline

if __name__ == "__main__":
    source = "videos/person_falls_videos.mp4"

    pipeline = VideoPipeline(source)
    pipeline.process()  