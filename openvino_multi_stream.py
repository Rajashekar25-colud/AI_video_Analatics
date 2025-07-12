
import cv2
import time
import multiprocessing
from openvino.runtime import Core

# Paths
MODEL_XML = "models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
VIDEO_PATH = "video1.mp4"

# Number of parallel streams
NUM_STREAMS = 5

# Load model once globally
core = Core()
model = core.read_model(model=MODEL_XML)
compiled_model = core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def run_stream(stream_id):
    print(f"[Stream {stream_id}] Starting...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Stream {stream_id}] Failed to open video.")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_image = cv2.resize(frame, (input_layer.shape[3], input_layer.shape[2]))
        input_image = input_image.transpose((2, 0, 1))  # HWC → CHW
        input_image = input_image.reshape(1, 3, input_layer.shape[2], input_layer.shape[3])

        # Inference
        result = compiled_model([input_image])[output_layer]

        # Postprocess
        for detection in result[0][0]:
            conf = detection[2]
            if conf > 0.5:
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_count += 1
        cv2.imshow(f"Stream {stream_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"[Stream {stream_id}] FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Launching {NUM_STREAMS} parallel CPU streams...\n")
    processes = []
    for i in range(NUM_STREAMS):
        p = multiprocessing.Process(target=run_stream, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
