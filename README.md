\# Face Recognition Attendance System



A real-time face recognition attendance system built using Python, OpenCV, dlib, and the face-recognition library.



\## Features



\- Real-time face detection

\- 128-dimensional face embeddings

\- Similarity-based face matching

\- Threshold-based recognition

\- Unknown-person rejection

\- Automatic attendance marking

\- Unknown-person photo capture

\- Timestamped unknown-person images

\- CSV-based attendance records



\## System Workflow



Camera Frame

↓

Face Detection

↓

Face Embedding Generation

↓

Face Distance Calculation

↓

Threshold-Based Matching

↓

Known Person / Unknown Person

↓

Attendance / Unknown Photo Capture



\## Model Used



The system uses the `face\_recognition` Python library with dlib-based face detection and face recognition.



Each detected face is converted into a 128-dimensional numerical face embedding.



\## Face Matching



The system calculates the face distance between the live face embedding and the stored known-face embedding.



The best match is selected using the minimum face distance.



\### Matching Threshold



Current threshold:



`0.50`



Decision rule:



\- Distance ≤ 0.50 → Known person

\- Distance > 0.50 → Unknown person



A lower face distance indicates greater similarity between the two face embeddings.



\## Unknown Rejection



If the best face distance is greater than the threshold, the system rejects the face as unknown.



For an unknown person, the system:



1\. Displays `UNKNOWN DETECTED`

2\. Prints the face distance

3\. Captures the camera frame

4\. Saves the image with a timestamp

5\. Stores the image locally in `unknown\_faces/`



Personal face images and attendance records are excluded from the public repository.



\## Attendance



For a recognized person, the system records:



\- Name

\- Date

\- Time

\- Face distance

\- Recognition status



The system prevents repeated attendance marking for the same person on the same day.



\## Basic Evaluation



Functional tests were performed during development.



| Test | Observed Result |

|---|---|

| Known-person recognition | Successfully recognized as Basavaraj |

| Known-person face distance | 0.4365 |

| Recognition threshold | 0.50 |

| Unknown-person rejection | Successfully rejected as Unknown |

| Unknown-person face distance | 0.6898 |

| Unknown photo capture | Successfully saved with timestamp |



These are functional validation results from development testing, not statistically measured model accuracy.



A larger evaluation dataset should be used to calculate formal accuracy, precision, recall, false acceptance rate, and false rejection rate.



\## Failure Cases



The system may produce incorrect results in situations such as:



\- Poor lighting

\- Face partially covered by a mask or object

\- Large changes in face pose

\- Very small faces in the camera frame

\- Motion blur

\- Poor camera quality

\- Multiple people with similar facial features

\- Significant changes in appearance

\- Insufficient reference images for a person



\## Possible Improvements



Future improvements include:



\- Add multiple reference images for each person

\- Perform a larger quantitative evaluation

\- Tune the matching threshold using validation data

\- Add multiple-person attendance handling

\- Improve low-light image processing

\- Add a database instead of CSV storage

\- Add a graphical user interface

\- Add anti-spoofing / liveness detection

\- Improve face detection for difficult poses

\- Add secure user authentication and access control



\## Project Structure



```text

Face\_Recognition\_Attendance\_System/

│

├── main.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── known\_faces/

│   └── Basavaraj.jpg        # kept local, not uploaded

│

├── unknown\_faces/           # kept local, not uploaded

│

└── attendance.csv           # kept local, not uploaded

