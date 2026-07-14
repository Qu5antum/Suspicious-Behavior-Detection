from .pipline.video_pipeline import VideoPipeline

if __name__ == "__main__":
    source = 0

    pipeline = VideoPipeline(source)
    pipeline.process()  