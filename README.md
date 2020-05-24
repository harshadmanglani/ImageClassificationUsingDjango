# Image Classification using Django

This is an extension of the dog breed classification project. It can be used as a template for other deep learning image classification projects.

Here's how to use this code:
1. git clone to your computer.
2. Keep your deep learning model in .h5 format.
3. Edit deeplearningsettings.py with the following changes:

->if you trained your model with input images of 150*150 pixels, assign 150 to self.pixels (note - only square pixelmaps are supported for now)

->assign the path to your local model to self.path

->assign your output classes to self.output_classes in the following format - [(0, 'class_0'), (1, 'class_1')]
