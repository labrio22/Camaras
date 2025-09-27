import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd
import cv2
from ultralytics import YOLO
import math
import subprocess

def expand_zone(zone, frame_shape, expand):
    x1, y1, x2, y2 = zone
    x1 = max(0, x1 - expand)
    y1 = max(0, y1 - expand)
    x2 = min(frame_shape[1], x2 + expand)
    y2 = min(frame_shape[0], y2 + expand)
    return (x1, y1, x2, y2)

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def split_video(input_path, max_size_mb=200):
    total_size = os.path.getsize(input_path) / (1024 * 1024)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    chunks = math.ceil(total_size / max_size_mb)
    chunk_duration = duration / chunks

    output_files = []
    for i in range(chunks):
        start = i * chunk_duration
        output_file = f"chunk_{i+1}.mp4"
        cmd = [
            'ffmpeg',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(chunk_duration),
            '-c', 'copy',
            output_file,
            '-y'
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_files.append(output_file)
    return output_files

def analizar_video(video_path, start_time_str):
    import shutil

    # Limpia carpeta y crea de nuevo para borrar eventos previos
    if os.path.exists("detected_events"):
        shutil.rmtree("detected_events")
    os.makedirs("detected_events", exist_ok=True)

    # Resto de tu código sigue acá...

    model = YOLO("yolov8n.pt")
    output_dir = "detected_events"
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.strptime(start_time_str, "%H:%M:%S")

    auto_zone = (300, 200, 500, 400)
    expand_px = 300

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    events = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % 30 != 0:
            frame_index += 1
            continue

        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        auto_zone_expanded = expand_zone(auto_zone, frame.shape, expand_px)
        auto_center = box_center(auto_zone_expanded)

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if label == "person" and conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_box = (x1, y1, x2, y2)
                person_center = box_center(person_box)

                dist = distance(auto_center, person_center)

                if dist < 300:
                    seconds = int(frame_index / fps)
                    timestamp_real = start_time + timedelta(seconds=seconds)
                    timestamp_str = timestamp_real.strftime("%H:%M:%S")

                    filename = f"{output_dir}/event_{os.path.basename(video_path).replace('.mp4','')}_{frame_index}_{len(events)}.jpg"

                    cv2.rectangle(frame, auto_zone_expanded[:2], auto_zone_expanded[2:], (255, 0, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imwrite(filename, frame)

                    events.append({
                        "video": os.path.basename(video_path),
                        "frame": frame_index,
                        "time": timestamp_str,
                        "confidence": conf,
                        "distance": dist,
                        "file": filename
                    })
                    st.write(f"Detectado en frame {frame_index} a las {timestamp_str}, distancia {dist:.1f}px")

        frame_index += 1

    cap.release()
    df = pd.DataFrame(events)
    return df

def main():
    st.title("Análisis de acercamiento a auto")

    uploaded_file = st.file_uploader("Subí tu video (mp4, h264)", type=["mp4", "h264"])

    if uploaded_file is not None:
        os.makedirs("uploads", exist_ok=True)
        input_path = f"uploads/{uploaded_file.name}"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Archivo '{uploaded_file.name}' subido correctamente.")

        start_time_str = st.text_input("Ingresa la hora de inicio del video (HH:MM:SS)", "15:30:00")

        if st.button("Ejecutar análisis"):
            size_mb = os.path.getsize(input_path) / (1024 * 1024)
            st.write(f"Tamaño del video: {size_mb:.2f} MB")

            if size_mb > 200:
                st.write("Video muy pesado, dividiendo en fragmentos...")
                fragments = split_video(input_path, max_size_mb=200)
            else:
                fragments = [input_path]

            all_events = []
            for frag in fragments:
                st.write(f"Analizando fragmento {frag}...")
                df = analizar_video(frag, start_time_str)
                all_events.append(df)

            if all_events:
                df_total = pd.concat(all_events, ignore_index=True)
                df_total.to_csv("eventos_detectados.csv", index=False)
                st.success("Análisis terminado.")

                st.subheader("Eventos detectados")
                st.dataframe(df_total)

                st.subheader("Imágenes detectadas")
                imgs = sorted(os.listdir("detected_events"))
                for img_name in imgs:
                    img_path = os.path.join("detected_events", img_name)
                    st.image(img_path, caption=img_name)
            else:
                st.write("No se detectaron eventos.")

if __name__ == "__main__":
    main()
