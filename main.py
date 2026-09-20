import cv2
import face_recognition
import os
import csv
from datetime import datetime
import re

# ==============================
# CONFIGURATION
# ==============================

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
ATTENDANCE_FILE = "attendance.csv"

MATCH_THRESHOLD = 0.50

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)


# ==============================
# LOAD KNOWN FACES
# ==============================

known_encodings = []
known_names = []


def load_known_faces():
    encodings = []
    names = []

    for filename in os.listdir(KNOWN_FACES_DIR):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) != 1:
            print(f"Skipping {filename}: expected exactly one face.")
            continue

        face_encoding = face_recognition.face_encodings(
            image, face_locations
        )[0]

        name = os.path.splitext(filename)[0]

        encodings.append(face_encoding)
        names.append(name)

        print(f"Loaded: {name}")

    return encodings, names


known_encodings, known_names = load_known_faces()

print("----------------------------------------")
print("FACE RECOGNITION ATTENDANCE SYSTEM")
print("----------------------------------------")
print("Model: dlib face embeddings")
print("Embedding: 128-dimensional")
print(f"Matching threshold: {MATCH_THRESHOLD}")
print(f"Known persons: {known_names}")
print("----------------------------------------")
print("Press E = Enroll new person")
print("Press Q = Quit")
print("----------------------------------------")


# ==============================
# ATTENDANCE FUNCTION
# ==============================

def mark_attendance(name, distance):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)

    already_marked = False

    if file_exists:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if (
                    row.get("Name") == name
                    and row.get("Date") == today
                ):
                    already_marked = True
                    break

    if not already_marked:

        with open(
            ATTENDANCE_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Face_Distance",
                        "Status"
                    ]
                )

            writer.writerow(
                [
                    name,
                    today,
                    current_time,
                    f"{distance:.3f}",
                    "Recognized"
                ]
            )

        print(f"ATTENDANCE MARKED: {name}")

    return not already_marked


# ==============================
# CAMERA
# ==============================

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():

    print("ERROR: Could not open camera.")
    exit()


unknown_saved = False


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = video_capture.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for face_location, face_encoding in zip(
        face_locations,
        face_encodings
    ):

        top, right, bottom, left = face_location

        name = "Unknown"
        best_distance = None

        # --------------------------
        # MATCH AGAINST KNOWN FACES
        # --------------------------

        if len(known_encodings) > 0:

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_index = distances.argmin()

            best_distance = distances[best_index]

            if best_distance <= MATCH_THRESHOLD:

                name = known_names[best_index]

        # --------------------------
        # KNOWN PERSON
        # --------------------------

        if name != "Unknown":

            label = f"{name} | distance: {best_distance:.3f}"

            mark_attendance(
                name,
                best_distance
            )

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # --------------------------
        # UNKNOWN PERSON
        # --------------------------

        else:

            print("UNKNOWN DETECTED")

            if best_distance is not None:
                print(
                    f"Face distance: {best_distance:.3f}"
                )

            if not unknown_saved:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                unknown_filename = (
                    f"unknown_{timestamp}.jpg"
                )

                unknown_path = os.path.join(
                    UNKNOWN_FACES_DIR,
                    unknown_filename
                )

                cv2.imwrite(
                    unknown_path,
                    frame
                )

                print(
                    f"Unknown face saved: {unknown_path}"
                )

                unknown_saved = True

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "UNKNOWN",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # ==============================
    # DISPLAY INFORMATION
    # ==============================

    cv2.putText(
        frame,
        "E = Enroll | Q = Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Threshold: {MATCH_THRESHOLD}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ==============================
    # ENROLL NEW PERSON
    # ==============================

    if key == ord("e"):

        print("\n================================")
        print("ENROLL NEW PERSON")
        print("================================")

        if len(face_locations) != 1:

            print(
                "Enrollment failed: "
                "exactly ONE face must be visible."
            )

            continue

        person_name = input(
            "Enter person's name: "
        ).strip()

        # Remove unsafe filename characters
        person_name = re.sub(
            r'[<>:"/\\|?*]',
            "",
            person_name
        )

        if person_name == "":
            print("Invalid name.")
            continue

        # Capture a fresh frame
        ret, enrollment_frame = video_capture.read()

        if not ret:
            print("Could not capture enrollment image.")
            continue

        enrollment_rgb = cv2.cvtColor(
            enrollment_frame,
            cv2.COLOR_BGR2RGB
        )

        enrollment_locations = (
            face_recognition.face_locations(
                enrollment_rgb
            )
        )

        if len(enrollment_locations) != 1:

            print(
                "Enrollment failed: "
                "exactly one face must be visible."
            )

            continue

        filename = f"{person_name}.jpg"

        save_path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        cv2.imwrite(
            save_path,
            enrollment_frame
        )

        # Generate embedding immediately
        new_encoding = face_recognition.face_encodings(
            enrollment_rgb,
            enrollment_locations
        )[0]

        known_encodings.append(new_encoding)
        known_names.append(person_name)

        print("--------------------------------")
        print("ENROLLMENT SUCCESSFUL")
        print(f"Name: {person_name}")
        print(f"Saved: {save_path}")
        print("Face embedding generated.")
        print("--------------------------------")

        unknown_saved = False

    # ==============================
    # QUIT
    # ==============================

    elif key == ord("q"):

        break


# ==============================
# CLEANUP
# ==============================

video_capture.release()
cv2.destroyAllWindows()

print("System closed.")