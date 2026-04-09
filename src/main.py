from .pipeline import VideoPipeline

if __name__ == "__main__":
    source = 0 #"videos/video2.mp4"

    pipeline = VideoPipeline(source)
    pipeline.process()  