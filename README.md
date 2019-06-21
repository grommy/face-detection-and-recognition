## Face recognition experiments using opencv and neural nets
 
## Prerequisites:
* Gather train dataset (photos)
Structure dataset in a following way:
```
data/photos/
├── Person_1
│   ├── Screen\ Shot\ 2019-03-04\ at\ 9.39.34\ AM.png
│   ├── Screen\ Shot\ 2019-03-04\ at\ 9.39.51\ AM.png
...
├── Person_2
│   ├── DSC_0462-3.jpg
│   ├── DSC_0463-2.jpg

```
* unix-like os
* python3 and pip installed
* Install libs 
   - install [opencv](https://medium.com/@nuwanprabhath/installing-opencv-in-macos-high-sierra-for-python-3-89c79f0a246a)
   - install [face-regognition lib](https://github.com/ageitgey/face_recognition#installation-options)
   - install other requirements `pip install -r requirements.txt`

## To run:

* encode known faces to vectors: <br>
`python encode_faces.py --dataset data/photos/ -e data/encodings/face_vec.pkl -d 'dnn'`
* run main script
`python recognize_faces.py --detection 'dnn' --encodings data/encodings/face_vec.pkl --input 'live' --display 1`