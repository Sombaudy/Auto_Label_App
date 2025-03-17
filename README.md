# Auto Labeling Application

This was created for the purpose of generating new datasets for YOLO object detection models when you only have access to a scarce amount of data.

A pre-trained custom YOLO model trained on the scarce dataset can be used to run inference on new images and save their labels as data for a new dataset. Unsatisfactory results can be discarded or used for manual labeling if so desired.

The application is run on streamlit. To run it, type the console command

```streamlit run app.py```
