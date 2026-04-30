from .pipeline import VideoPipeline

if __name__ == "__main__":
    source = "videos/video456.mp4"

    pipeline = VideoPipeline(source)
    pipeline.process()  