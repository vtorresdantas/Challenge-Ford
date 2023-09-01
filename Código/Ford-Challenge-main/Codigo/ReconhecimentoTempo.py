import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

webcam = cv2.VideoCapture(0)
start_time = time.time()

while webcam.isOpened():
    success, img = webcam.read()
    
    # Aplicando o modelo FaceMesh usando o MediaPipe
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.FaceMesh(refine_landmarks=True).process(img)
    
    # Desenhar anotações na imagem
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    elapsed_time = time.time() - start_time
    elapsed_time_str = f"Tempo decorrido: {elapsed_time:.2f} segundos"
    cv2.putText(img, elapsed_time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Koolac", img)
    
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
