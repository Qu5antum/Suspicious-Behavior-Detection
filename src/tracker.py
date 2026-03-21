from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    """
    Videoda insanları takip etme sınıfı,
    Onaylanmış her bir parça için benzersiz bir kimlik ve koordinatlar (x1, y1, x2, y2) içeren sözlüklerden oluşan bir liste döndürür.
    """
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.3
        )
    """
    Güncelleme yöntemi bir çerçeve ve bir tespit listesi alır
    """
    def update(self, frame, detections):
        """
        detections format:
        [[x, y, w, h], confidence]
        """

        dets = [
            (det["bbox"], det["confidence"], "person")
            for det in detections
        ]

        tracks = self.tracker.update_tracks(dets, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            results.append({
                "id": track_id,
                "bbox": [l, t, r, b]
            })

        return results