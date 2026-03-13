import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path

print("=" * 50)
print("  Reconocimiento Facial")
print("=" * 50)
print()

# Ruta de la carpeta de rostros
FACES_DIR = Path(__file__).parent / "faces"

# Datos de rostros registrados
known_face_encodings = []
known_face_names = []
known_face_photos = []

# Cargar todos los rostros desde la carpeta faces
print(f"Cargando datos de rostros... ({FACES_DIR})")
print()

for file in FACES_DIR.iterdir():
    if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        name = file.stem
        print(f"  Cargando: {file.name} -> {name}")

        try:
            image = face_recognition.load_image_file(str(file))
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

                photo = cv2.imread(str(file))
                photo = cv2.resize(photo, (80, 80))
                known_face_photos.append(photo)
                print(f"    -> OK!")
            else:
                print(f"    -> No se detecto rostro")
        except Exception as e:
            print(f"    -> Error: {e}")

print()
print(f"Total {len(known_face_names)} personas registradas: {', '.join(known_face_names)}")
print()

# Imagen para Desconocido
unknown_photo = np.zeros((80, 80, 3), dtype=np.uint8)
unknown_photo[:] = (60, 60, 60)
cv2.putText(unknown_photo, "?", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 2)

# Abrir la camara
print("Iniciando camara...")
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

import time
time.sleep(1)

if not video_capture.isOpened():
    print()
    print("Error: No se pudo abrir la camara!")
    print()
    print("Solucion:")
    print("1. Configuracion del sistema -> Privacidad y seguridad -> Camara")
    print("2. Permitir Terminal (o la aplicacion en uso)")
    print("3. Ejecutar de nuevo")
    exit(1)

print("Camara iniciada con exito!")
print()
print("Instrucciones:")
print("  - Mira hacia la camara")
print("  - Haz clic en el boton SALIR para terminar")
print()

# Bandera de salida
should_quit = False

# Procesamiento de clic del raton
def mouse_callback(event, x, y, flags, param):
    global should_quit, best_matches
    if event == cv2.EVENT_LBUTTONDOWN:
        panel_x = x - 640
        # Boton SALIR
        if 250 <= panel_x <= 350 and 435 <= y <= 470:
            should_quit = True
        # Boton REINICIAR
        elif 20 <= panel_x <= 130 and 435 <= y <= 470:
            best_matches = {}

cv2.namedWindow('Reconocimiento Facial')
cv2.setMouseCallback('Reconocimiento Facial', mouse_callback)

# Contador de cuadros
frame_count = 0
process_every_n_frames = 2

# Resultados de reconocimiento actuales
current_results = []

# Mantener la mejor coincidencia para cada persona (registrar el porcentaje mas alto con el nombre como clave)
best_matches = {}

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % process_every_n_frames == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        current_results = []

        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = [v * 2 for v in face_location]

            landmarks = None
            if i < len(face_landmarks_list):
                landmarks = {}
                for feature, points in face_landmarks_list[i].items():
                    landmarks[feature] = [(p[0] * 2, p[1] * 2) for p in points]

            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                match_percent = max(0, min(100, (1 - best_distance) * 100 * 1.5))

                if best_distance < 0.5:
                    name = known_face_names[best_match_index]
                    photo = known_face_photos[best_match_index]
                    is_match = True
                elif best_distance < 0.6:
                    name = known_face_names[best_match_index] + "?"
                    photo = known_face_photos[best_match_index]
                    is_match = True
                else:
                    name = "Desconocido"
                    photo = unknown_photo
                    is_match = False
                    match_percent = max(0, (1 - best_distance) * 100)
            else:
                name = "Desconocido"
                photo = unknown_photo
                is_match = False
                match_percent = 0

            # Recortar el rostro
            padding = 20
            face_top = max(0, top - padding)
            face_bottom = min(frame.shape[0], bottom + padding)
            face_left = max(0, left - padding)
            face_right = min(frame.shape[1], right + padding)
            captured_face = frame[face_top:face_bottom, face_left:face_right].copy()

            if captured_face.size > 0:
                captured_face = cv2.resize(captured_face, (80, 80))
            else:
                captured_face = None

            # Actualizar la mejor coincidencia (mantener el porcentaje mas alto por nombre)
            base_name = name.replace("?", "")
            if base_name not in best_matches or match_percent > best_matches[base_name]['percent']:
                best_matches[base_name] = {
                    'percent': match_percent,
                    'captured_face': captured_face,
                    'landmarks': landmarks,
                    'photo': photo,
                    'is_match': is_match,
                    'name': name
                }

            current_results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'photo': photo,
                'percent': match_percent,
                'is_match': is_match,
                'landmarks': landmarks,
                'captured_face': captured_face,
                'person_id': i + 1,
                'base_name': base_name
            })

    # Colores para cada persona (hasta 6 personas)
    person_colors = [
        (0, 255, 0),    # Verde
        (255, 165, 0),  # Naranja
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
        (255, 0, 0),    # Azul
        (0, 165, 255),  # Naranja claro
    ]

    # Dibujar
    for result in current_results:
        top, right, bottom, left = result['location']
        name = result['name']
        percent = result['percent']
        is_match = result['is_match']
        landmarks = result.get('landmarks')
        person_id = result.get('person_id', 1)

        # Color segun persona (si no es match, usar rojo)
        if is_match:
            color = person_colors[(person_id - 1) % len(person_colors)]
        else:
            color = (0, 0, 255)

        # Marco del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Mostrar numero de persona en la esquina superior izquierda
        cv2.circle(frame, (left + 15, top + 15), 12, color, -1)
        cv2.putText(frame, str(person_id), (left + 10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Dibujar puntos de referencia con el mismo color de la persona
        if landmarks:
            if 'chin' in landmarks:
                points = landmarks['chin']
                for j in range(len(points) - 1):
                    cv2.line(frame, points[j], points[j + 1], color, 1)
            for brow in ['left_eyebrow', 'right_eyebrow']:
                if brow in landmarks:
                    points = landmarks[brow]
                    for j in range(len(points) - 1):
                        cv2.line(frame, points[j], points[j + 1], color, 1)
            for eye in ['left_eye', 'right_eye']:
                if eye in landmarks:
                    points = landmarks[eye]
                    pts = np.array(points, np.int32)
                    cv2.polylines(frame, [pts], True, color, 1)
            if 'nose_bridge' in landmarks:
                points = landmarks['nose_bridge']
                for j in range(len(points) - 1):
                    cv2.line(frame, points[j], points[j + 1], color, 1)
            if 'nose_tip' in landmarks:
                points = landmarks['nose_tip']
                for j in range(len(points) - 1):
                    cv2.line(frame, points[j], points[j + 1], color, 1)
            for lip in ['top_lip', 'bottom_lip']:
                if lip in landmarks:
                    points = landmarks[lip]
                    pts = np.array(points, np.int32)
                    cv2.polylines(frame, [pts], True, color, 1)

        # Etiqueta de nombre
        label = f"{name} {percent:.0f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (left, bottom), (left + label_size[0] + 10, bottom + 25), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 5, bottom + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Crear panel lateral
    panel_height = frame.shape[0]
    panel_width = 360
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Titulo - mostrar personas registradas (de best_matches)
    registered_people = [k for k in best_matches.keys() if k != "Desconocido"]
    num_registered = len(registered_people)
    title = f"REGISTRADOS: {num_registered} persona(s)"
    cv2.putText(panel, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.line(panel, (20, 40), (panel_width - 20, 40), (80, 80, 80), 1)

    # Mostrar informacion de cada persona desde best_matches
    y_offset = 55
    max_people = 4  # Mostrar maximo 4 personas

    for idx, base_name in enumerate(registered_people[:max_people]):
        person_id = idx + 1
        data = best_matches[base_name]
        name = data.get('name', base_name)
        is_match = data.get('is_match', True)
        registered_photo = data.get('photo', unknown_photo)
        percent = data['percent']
        captured_face = data['captured_face']

        # Fondo de la seccion de persona
        section_height = 95
        cv2.rectangle(panel, (10, y_offset), (panel_width - 10, y_offset + section_height), (45, 45, 45), -1)

        # Color segun persona
        if is_match:
            color = person_colors[(person_id - 1) % len(person_colors)]
        else:
            color = (0, 0, 255)
        cv2.circle(panel, (30, y_offset + 15), 10, color, -1)
        cv2.putText(panel, str(person_id), (25, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Nombre y probabilidad
        display_name = name[:12] if len(name) > 12 else name
        cv2.putText(panel, display_name, (50, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(panel, f"{percent:.0f}%", (panel_width - 60, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Foto registrada
        cv2.putText(panel, "Reg.", (20, y_offset + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        panel[y_offset + 42:y_offset + 42 + 50, 20:20 + 50] = cv2.resize(registered_photo, (50, 50))
        cv2.rectangle(panel, (20, y_offset + 42), (70, y_offset + 92), (100, 100, 100), 1)

        # VS
        cv2.putText(panel, "vs", (80, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Rostro capturado
        cv2.putText(panel, "Cap.", (100, y_offset + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        if captured_face is not None:
            panel[y_offset + 42:y_offset + 42 + 50, 100:100 + 50] = cv2.resize(captured_face, (50, 50))
            cv2.rectangle(panel, (100, y_offset + 42), (150, y_offset + 92), color, 1)
        else:
            cv2.rectangle(panel, (100, y_offset + 42), (150, y_offset + 92), (60, 60, 60), -1)

        # Barra de progreso
        bar_x = 165
        bar_width_max = 170
        cv2.rectangle(panel, (bar_x, y_offset + 55), (bar_x + bar_width_max, y_offset + 75), (60, 60, 60), -1)
        bar_fill = int(bar_width_max * percent / 100)
        if bar_fill > 0:
            bar_color = (0, 255, 0) if percent > 60 else (0, 255, 255) if percent > 40 else (0, 100, 255)
            cv2.rectangle(panel, (bar_x, y_offset + 55), (bar_x + bar_fill, y_offset + 75), bar_color, -1)
        cv2.rectangle(panel, (bar_x, y_offset + 55), (bar_x + bar_width_max, y_offset + 75), (100, 100, 100), 1)

        # Estado de deteccion de puntos de referencia (indicadores pequenos) - usar mejor coincidencia
        if base_name in best_matches:
            landmarks = best_matches[base_name].get('landmarks')
        else:
            landmarks = result.get('landmarks')
        indicator_x = 165
        cv2.putText(panel, "Detectado:", (indicator_x, y_offset + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        if landmarks:
            indicators = [
                ('chin', (255, 200, 0)),
                ('left_eye', (0, 255, 0)),
                ('nose_tip', (255, 0, 255)),
                ('top_lip', (0, 0, 255)),
            ]
            for j, (key, ind_color) in enumerate(indicators):
                if key in landmarks:
                    cv2.circle(panel, (indicator_x + 60 + j * 18, y_offset + 85), 5, ind_color, -1)
                else:
                    cv2.circle(panel, (indicator_x + 60 + j * 18, y_offset + 85), 5, (60, 60, 60), -1)

        y_offset += section_height + 5

    # Si no se detectan personas
    if len(current_results) == 0:
        cv2.putText(panel, "Sin rostro detectado", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(panel, "Mira hacia la camara", (75, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Leyenda
    legend_y = panel_height - 80
    cv2.line(panel, (20, legend_y - 10), (panel_width - 20, legend_y - 10), (60, 60, 60), 1)
    cv2.putText(panel, "Analisis:", (20, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    legend_items = [
        ("Contorno", (255, 200, 0)),
        ("Ojos", (0, 255, 0)),
        ("Nariz", (255, 0, 255)),
        ("Labios", (0, 0, 255)),
    ]
    legend_x = 90
    for item_name, item_color in legend_items:
        cv2.circle(panel, (legend_x, legend_y + 2), 4, item_color, -1)
        cv2.putText(panel, item_name, (legend_x + 8, legend_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        legend_x += 65

    # Boton REINICIAR
    cv2.rectangle(panel, (20, panel_height - 45), (130, panel_height - 10), (100, 100, 0), -1)
    cv2.rectangle(panel, (20, panel_height - 45), (130, panel_height - 10), (150, 150, 0), 2)
    cv2.putText(panel, "REINICIAR", (25, panel_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Boton SALIR
    btn_x1, btn_y1 = 250, panel_height - 45
    btn_x2, btn_y2 = 340, panel_height - 10
    cv2.rectangle(panel, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 150), -1)
    cv2.rectangle(panel, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), 2)
    cv2.putText(panel, "SALIR", (btn_x1 + 15, btn_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combinar y mostrar
    combined = np.hstack((frame, panel))
    cv2.imshow('Reconocimiento Facial', combined)

    # Procesar teclas
    cv2.waitKey(1)

    if should_quit:
        break

video_capture.release()
cv2.destroyAllWindows()
print()
print("Finalizado")
