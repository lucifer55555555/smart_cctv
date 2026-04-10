import cv2

from app.pipeline import CampusSafetyPipeline


def main() -> None:
    """
    Local runner for the campus safety system without Flask web UI.

    Shows an annotated video window from the campus camera.
    Press 'q' to quit.
    """
    pipeline = CampusSafetyPipeline()
    try:
        for ok, frame, risk in pipeline.frames():
            if not ok or frame is None:
                print("[main] No frame from campus camera, exiting.")
                break

            cv2.imshow("Campus Safety Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

