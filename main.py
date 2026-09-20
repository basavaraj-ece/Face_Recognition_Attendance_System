import cv2
import face_recognition
import csv
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

MATCH_THRESHOLD = 0.50

KNOWN_FACE_PATH = "known_faces/Basavaraj.jpg"
ATTENDANCE_FILE = "attendance.csv"
UNKNOWN_FOLDER = "unknown_faces"

# Create unknown folder if it doesn't exist
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)


# ============================================================
# LOAD KNOWN FACE
# ============================================================

print("Loading known face...")

known_image = face_recognition.load_image_file(KNOWN_FACE_PATH)

known_encodings = face_recognition.face_encodings(known_image)

if len(known_encodings) == 0:
    raise RuntimeError(
        "No face detected in known_faces/Basavaraj.jpg"
    )

known_encoding = known_encodings[0]

known_face_encodings = [known_encoding]
known_face_names = ["Basavaraj"]

print("Known face embedding created successfully.")


# ============================================================
# CREATE ATTENDANCE FILE
# ============================================================

if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:

    with open(ATTENDANCE_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time",
            "Face_Distance",
            "Status"
        ])


# Prevent repeated attendance
marked_today = set()

# Prevent saving unknown photos continuously
unknown_saved = False


# ============================================================
# START CAMERA
# ============================================================

video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Could not access the camera.")

print()
print("==========================================")
print(" FACE RECOGNITION ATTENDANCE SYSTEM")
print("==========================================")
print("Model       : dlib / face_recognition")
print("Embedding   : 128-dimensional face encoding")
print("Threshold   :", MATCH_THRESHOLD)
print("Unknown     : Photo capture enabled")
print("Press Q to quit")
print("==========================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = video.read()

    if not ret:
        print("Could not read camera frame.")
        break

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )

    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )

    # --------------------------------------------------------
    # FACE EMBEDDINGS
    # --------------------------------------------------------

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )


    # --------------------------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------------------------

    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        # ----------------------------------------------------
        # FACE DISTANCE
        # ----------------------------------------------------

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        best_match_index = face_distances.argmin()

        best_distance = face_distances[best_match_index]


        # ----------------------------------------------------
        # UNKNOWN REJECTION
        # ----------------------------------------------------

        if best_distance <= MATCH_THRESHOLD:

            name = known_face_names[best_match_index]
            status = "Recognized"

        else:

            name = "Unknown"
            status = "Rejected"


        # ----------------------------------------------------
        # SCALE COORDINATES
        # ----------------------------------------------------

        top, right, bottom, left = face_location

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # ----------------------------------------------------
        # UNKNOWN PERSON CAPTURE
        # ----------------------------------------------------

        if name == "Unknown":

            box_color = (0, 0, 255)

            if not unknown_saved:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                filename = (
                    f"unknown_{timestamp}.jpg"
                )

                filepath = os.path.join(
                    UNKNOWN_FOLDER,
                    filename
                )

                # Save complete camera frame
                cv2.imwrite(
                    filepath,
                    frame
                )

                print()
                print("================================")
                print("UNKNOWN DETECTED")
                print("Photo saved:", filepath)
                print("Face distance:", round(best_distance, 4))
                print("================================")
                print()

                unknown_saved = True

        else:

            box_color = (0, 255, 0)


        # ----------------------------------------------------
        # DRAW FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            box_color,
            2
        )


        # ----------------------------------------------------
        # DISPLAY NAME + DISTANCE
        # ----------------------------------------------------

        label = f"{name} | distance: {best_distance:.3f}"

        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            box_color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )


        # ----------------------------------------------------
        # ATTENDANCE
        # ----------------------------------------------------

        if name != "Unknown":

            today = datetime.now().strftime(
                "%Y-%m-%d"
            )

            attendance_key = name + "_" + today

            if attendance_key not in marked_today:

                current_time = datetime.now().strftime(
                    "%H:%M:%S"
                )

                with open(
                    ATTENDANCE_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        name,
                        today,
                        current_time,
                        f"{best_distance:.4f}",
                        status
                    ])

                marked_today.add(attendance_key)

                print(
                    f"Attendance marked: "
                    f"{name} | "
                    f"Distance: {best_distance:.4f}"
                )


    # --------------------------------------------------------
    # SYSTEM INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Threshold: {MATCH_THRESHOLD}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

video.release()
cv2.destroyAllWindows()

print("System stopped.")